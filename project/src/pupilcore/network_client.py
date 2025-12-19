import zmq


class PupilCoreNetworkClient:
    def __init__(self, host="127.0.0.1", port=50020, timeout_ms=1000):
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self._context = zmq.Context.instance()
        self._req = None
        self._sub_port = None

    def connect(self):
        if self._req is None:
            self._req = self._context.socket(zmq.REQ)
            self._req.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
            self._req.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
            self._req.connect(f"tcp://{self.host}:{self.port}")
        try:
            self._req.send_string("SUB_PORT")
            self._sub_port = self._req.recv_string()
        except zmq.error.Again as exc:
            raise RuntimeError(
                "Pupil Remote did not respond. Ensure Pupil Capture is running, "
                "the Network API plugin is enabled, and the host/port are correct."
            ) from exc
        return self._sub_port

    def create_subscriber(self, topics):
        if self._sub_port is None:
            self.connect()
        sub = self._context.socket(zmq.SUB)
        sub.connect(f"tcp://{self.host}:{self._sub_port}")
        for topic in topics:
            sub.setsockopt_string(zmq.SUBSCRIBE, topic)
        return sub

    def close(self):
        if self._req is not None:
            self._req.close(0)
            self._req = None

        # Context is shared; do not terminate here.
