import threading

import cv2
import msgpack
import numpy as np
import zmq


class FrameReceiver(threading.Thread):
    def __init__(self, sub_socket, poll_timeout_ms=100):
        super().__init__(daemon=True)
        self._sub_socket = sub_socket
        self._poll_timeout_ms = poll_timeout_ms
        self._lock = threading.Lock()
        self._latest = None
        self._running = threading.Event()

    def run(self):
        self._running.set()
        poller = zmq.Poller()
        poller.register(self._sub_socket, zmq.POLLIN)
        while self._running.is_set():
            events = dict(poller.poll(self._poll_timeout_ms))
            if self._sub_socket not in events:
                continue
            parts = self._sub_socket.recv_multipart()
            if len(parts) < 2:
                continue
            topic, payload = parts[0], parts[1]
            if not topic.startswith(b"frame.world"):
                continue
            data = msgpack.unpackb(payload, raw=False)
            if len(parts) >= 3 and "data" not in data:
                data["data"] = parts[2]
            frame = self._decode_frame(data)
            if frame is None:
                continue
            with self._lock:
                self._latest = {
                    "frame": frame,
                    "timestamp": data.get("timestamp"),
                }

    def stop(self):
        self._running.clear()

    def get_latest(self):
        with self._lock:
            if self._latest is None:
                return None
            return {
                "frame": self._latest["frame"].copy(),
                "timestamp": self._latest.get("timestamp"),
            }

    def _decode_frame(self, data):
        width = data.get("width")
        height = data.get("height")
        frame_format = data.get("format")
        if isinstance(frame_format, bytes):
            frame_format = frame_format.decode("utf-8")
        buffer = data.get("data")
        if buffer is None:
            return None
        raw = np.frombuffer(buffer, dtype=np.uint8)
        if frame_format in ("jpeg", "jpg"):
            decoded = cv2.imdecode(raw, cv2.IMREAD_COLOR)
            return decoded
        if width is None or height is None:
            decoded = cv2.imdecode(raw, cv2.IMREAD_COLOR)
            return decoded
        if frame_format == "gray":
            img = raw.reshape((height, width))
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if frame_format == "bgr":
            return raw.reshape((height, width, 3))
        try:
            return raw.reshape((height, width, 3))
        except ValueError:
            return cv2.imdecode(raw, cv2.IMREAD_COLOR)
