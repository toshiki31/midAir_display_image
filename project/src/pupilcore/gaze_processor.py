import threading

import msgpack
import zmq


class GazeProcessor(threading.Thread):
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
            if not topic.startswith(b"gaze."):
                continue
            data = msgpack.unpackb(payload, raw=False)
            norm_pos = data.get("norm_pos")
            if norm_pos is None or len(norm_pos) != 2:
                continue
            gaze_x = float(norm_pos[0])
            gaze_y = 1.0 - float(norm_pos[1])
            gaze = {
                "timestamp": data.get("timestamp"),
                "x": gaze_x,
                "y": gaze_y,
                "confidence": float(data.get("confidence", 0.0)),
            }
            with self._lock:
                self._latest = gaze

    def stop(self):
        self._running.clear()

    def get_latest(self):
        with self._lock:
            if self._latest is None:
                return None
            return dict(self._latest)
