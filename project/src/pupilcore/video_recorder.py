import cv2


class VideoRecorder:
    def __init__(self, output_path, fps=30):
        self.output_path = output_path
        self.fps = fps
        self._writer = None

    def write(self, frame_bgr, gaze, faces, gaze_on_face=False, gaze_face=None):
        if frame_bgr is None:
            return
        if self._writer is None:
            height, width = frame_bgr.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, (width, height))
        annotated = frame_bgr.copy()
        self._draw_faces(annotated, faces, gaze_face)
        self._draw_gaze(annotated, gaze)
        self._writer.write(annotated)

    def close(self):
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    def _draw_faces(self, frame_bgr, faces, gaze_face):
        height, width = frame_bgr.shape[:2]
        for face in faces:
            left = int(face["left"] * width)
            top = int(face["top"] * height)
            right = int((face["left"] + face["width"]) * width)
            bottom = int((face["top"] + face["height"]) * height)
            if gaze_face is not None and face is gaze_face:
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)
            cv2.rectangle(frame_bgr, (left, top), (right, bottom), color, 2)

    def _draw_gaze(self, frame_bgr, gaze):
        if gaze is None:
            return
        gaze_x = gaze.get("x")
        gaze_y = gaze.get("y")
        if gaze_x is None or gaze_y is None:
            return
        height, width = frame_bgr.shape[:2]
        center = (int(gaze_x * width), int(gaze_y * height))
        confidence = float(gaze.get("confidence", 0.0))
        radius = int(5 + 10 * confidence)
        cv2.circle(frame_bgr, center, radius, (0, 0, 255), -1)
