import argparse
import csv
import os
import sys
import time
from datetime import datetime
from threading import Event, Thread

from pupilcore import (
    FaceDetector,
    FrameReceiver,
    GazeFaceAnalyzer,
    GazeProcessor,
    PupilCoreNetworkClient,
    VideoRecorder,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Record Pupil Core world camera with gaze and face overlays."
    )
    parser.add_argument("--output", default="output", help="Output directory root")
    parser.add_argument("--fps", type=float, default=30.0, help="Target FPS")
    parser.add_argument("--duration", type=float, default=0.0, help="Duration in seconds (0 for unlimited)")
    parser.add_argument(
        "--aws",
        action="store_true",
        default=True,
        help="Use AWS Rekognition for face detection (default: True)",
    )
    parser.add_argument(
        "--no-aws",
        dest="aws",
        action="store_false",
        help="Use OpenCV Haar Cascade instead of AWS Rekognition",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Pupil Remote host")
    parser.add_argument("--port", type=int, default=50020, help="Pupil Remote port")
    parser.add_argument(
        "--face-interval",
        type=int,
        default=5,
        help="Detect faces every N frames",
    )
    parser.add_argument(
        "--gaze-confidence-threshold",
        type=float,
        default=0.6,
        help="Confidence threshold to report calibration as OK",
    )
    parser.add_argument(
        "--aws-region",
        default=os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION"),
        help="AWS region for Rekognition",
    )
    return parser.parse_args()


def ensure_output_dirs(base_dir):
    videos_dir = os.path.join(base_dir, "videos")
    csv_dir = os.path.join(base_dir, "csv")
    os.makedirs(videos_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)
    return videos_dir, csv_dir


def write_csv_log(path, rows):
    fields = [
        "timestamp",
        "gaze_x",
        "gaze_y",
        "gaze_confidence",
        "num_faces",
        "is_looking_at_face",
        "face_bbox_left",
        "face_bbox_top",
        "face_bbox_width",
        "face_bbox_height",
    ]
    with open(path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path, summary):
    with open(path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["metric", "value"])
        for key, value in summary.items():
            writer.writerow([key, value])


def start_keyboard_listener(stop_event):
    def _run_raw():
        try:
            import termios
            import tty
        except ImportError:
            _run_line()
            return

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            while not stop_event.is_set():
                ch = sys.stdin.read(1)
                if ch.lower() == "q" or ch == "\x1b":
                    stop_event.set()
                    break
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def _run_line():
        while not stop_event.is_set():
            line = sys.stdin.readline().strip().lower()
            if line in {"q", "esc", "exit"}:
                stop_event.set()
                break

    thread = Thread(target=_run_raw, daemon=True)
    thread.start()
    return thread


def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    videos_dir, csv_dir = ensure_output_dirs(args.output)
    video_path = os.path.join(videos_dir, f"{timestamp}_pupilcore_gaze.mp4")
    log_path = os.path.join(csv_dir, f"{timestamp}_gaze_log.csv")
    summary_path = os.path.join(csv_dir, f"{timestamp}_summary.csv")

    print("Connecting to Pupil Capture...")
    client = PupilCoreNetworkClient(host=args.host, port=args.port)
    gaze_sub = client.create_subscriber(["gaze."])
    frame_sub = client.create_subscriber(["frame.world"])
    print("Subscribed to gaze and world frame streams.")

    gaze_processor = GazeProcessor(gaze_sub)
    frame_receiver = FrameReceiver(frame_sub)
    face_detector = FaceDetector(use_aws=args.aws, aws_region=args.aws_region)
    analyzer = GazeFaceAnalyzer()
    recorder = VideoRecorder(video_path, fps=args.fps)

    gaze_processor.start()
    frame_receiver.start()
    print("Recording started. Press 'q' or 'Esc' to stop.")

    log_rows = []
    total_frames = 0
    face_gaze_frames = 0
    face_present_frames = 0
    cached_faces = []
    stop_event = Event()
    start_keyboard_listener(stop_event)

    start_time = time.time()
    next_frame_time = start_time
    last_gaze_time = None
    last_frame_time = None
    frame_ready_logged = False
    gaze_ready_logged = False
    calibration_logged = False
    warned_no_frame = False
    warned_no_gaze = False

    try:
        while True:
            if stop_event.is_set():
                break
            now = time.time()
            if args.duration > 0 and (now - start_time) >= args.duration:
                break
            if now < next_frame_time:
                time.sleep(max(0.0, next_frame_time - now))
                continue
            next_frame_time += 1.0 / args.fps

            latest_frame = frame_receiver.get_latest()
            if latest_frame is None:
                if not warned_no_frame and (now - start_time) > 5:
                    print("Waiting for world frames... (check Pupil Capture world stream)")
                    warned_no_frame = True
                continue
            frame = latest_frame["frame"]
            last_frame_time = now
            if not frame_ready_logged:
                print("World frame stream active.")
                frame_ready_logged = True

            if total_frames % max(args.face_interval, 1) == 0:
                cached_faces = face_detector.detect(frame)
            faces = cached_faces

            gaze = gaze_processor.get_latest()
            if gaze is not None:
                last_gaze_time = now
                if not gaze_ready_logged:
                    print("Gaze stream active.")
                    gaze_ready_logged = True
                if not calibration_logged and gaze.get("confidence", 0.0) >= args.gaze_confidence_threshold:
                    print(
                        f"Calibration appears OK (confidence >= {args.gaze_confidence_threshold})."
                    )
                    calibration_logged = True
            elif not warned_no_gaze and (now - start_time) > 5:
                print("Waiting for gaze data... (check calibration and gaze mapper)")
                warned_no_gaze = True
            is_looking, face_bbox = analyzer.analyze(gaze, faces)
            if faces:
                face_present_frames += 1
            if is_looking:
                face_gaze_frames += 1

            recorder.write(frame, gaze, faces, is_looking, face_bbox)

            log_rows.append(
                {
                    "timestamp": now,
                    "gaze_x": "" if gaze is None else gaze.get("x"),
                    "gaze_y": "" if gaze is None else gaze.get("y"),
                    "gaze_confidence": "" if gaze is None else gaze.get("confidence"),
                    "num_faces": len(faces),
                    "is_looking_at_face": is_looking,
                    "face_bbox_left": "" if face_bbox is None else face_bbox.get("left"),
                    "face_bbox_top": "" if face_bbox is None else face_bbox.get("top"),
                    "face_bbox_width": "" if face_bbox is None else face_bbox.get("width"),
                    "face_bbox_height": "" if face_bbox is None else face_bbox.get("height"),
                }
            )

            total_frames += 1
    finally:
        end_time = time.time()
        gaze_processor.stop()
        frame_receiver.stop()
        gaze_processor.join(timeout=1.0)
        frame_receiver.join(timeout=1.0)
        gaze_sub.close(0)
        frame_sub.close(0)
        recorder.close()
        client.close()
        write_csv_log(log_path, log_rows)

        total_duration = max(0.0, end_time - start_time)
        summary = {
            "session_id": timestamp,
            "total_duration_sec": round(total_duration, 3),
            "total_frames": total_frames,
            "face_present_frames": face_present_frames,
            "face_present_duration_sec": round(face_present_frames / args.fps, 3)
            if args.fps > 0
            else 0.0,
            "face_gaze_frames": face_gaze_frames,
            "face_gaze_duration_sec": round(face_gaze_frames / args.fps, 3)
            if args.fps > 0
            else 0.0,
            "face_gaze_percentage": round(
                (face_gaze_frames / total_frames * 100.0) if total_frames else 0.0, 3
            ),
            "fps": args.fps,
        }
        write_summary(summary_path, summary)

        print("Recording finished")
        for key, value in summary.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
