class GazeFaceAnalyzer:
    def analyze(self, gaze, faces):
        if gaze is None:
            return False, None
        gaze_x = gaze.get("x")
        gaze_y = gaze.get("y")
        if gaze_x is None or gaze_y is None:
            return False, None

        for face in faces:
            left = face["left"]
            top = face["top"]
            width = face["width"]
            height = face["height"]
            expand_x = width * 0.10
            expand_y = height * 0.10
            left_expanded = max(0.0, left - expand_x)
            top_expanded = max(0.0, top - expand_y)
            right_expanded = min(1.0, left + width + expand_x)
            bottom_expanded = min(1.0, top + height + expand_y)
            if left_expanded <= gaze_x <= right_expanded and top_expanded <= gaze_y <= bottom_expanded:
                return True, face
        return False, None
