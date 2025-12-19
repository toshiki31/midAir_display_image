import cv2
import numpy as np

try:
    import boto3
except ImportError:  # pragma: no cover - optional AWS dependency
    boto3 = None


class FaceDetector:
    def __init__(
        self,
        use_aws=False,
        scale_factor=1.1,
        min_neighbors=5,
        aws_region=None,
    ):
        self.use_aws = use_aws
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.aws_region = aws_region
        self._cascade = None
        self._rekognition = None

        if self.use_aws:
            if boto3 is None:
                raise RuntimeError("boto3 is required for AWS face detection")
            self._rekognition = boto3.client("rekognition", region_name=self.aws_region)
        else:
            self._cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )

    def detect(self, frame_bgr):
        if self.use_aws:
            return self._detect_aws(frame_bgr)
        return self._detect_opencv(frame_bgr)

    def _detect_opencv(self, frame_bgr):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self._cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
        )
        height, width = frame_bgr.shape[:2]
        results = []
        for (x, y, w, h) in faces:
            results.append(
                {
                    "left": x / width,
                    "top": y / height,
                    "width": w / width,
                    "height": h / height,
                }
            )
        return results

    def _detect_aws(self, frame_bgr):
        ok, buf = cv2.imencode(".jpg", frame_bgr)
        if not ok:
            return []
        response = self._rekognition.detect_faces(
            Image={"Bytes": buf.tobytes()},
            Attributes=["ALL"],
        )
        results = []
        for face in response.get("FaceDetails", []):
            bbox = face.get("BoundingBox", {})
            results.append(
                {
                    "left": float(bbox.get("Left", 0.0)),
                    "top": float(bbox.get("Top", 0.0)),
                    "width": float(bbox.get("Width", 0.0)),
                    "height": float(bbox.get("Height", 0.0)),
                }
            )
        return results
