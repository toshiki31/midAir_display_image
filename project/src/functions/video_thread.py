import logging
import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

# 定数
WINDOW_WIDTH = 2500
WINDOW_HEIGHT = int(WINDOW_WIDTH * 10 / 16)

# ロガーセットアップ
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoThread(QThread):
    frame_ready = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.running = False
        self.video_path = None

    def start_video(self, video_path):
        """動画再生を開始"""
        self.video_path = video_path
        if not self.running:
            self.running = True
            self.start()

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            logger.error(f"Error: Cannot open video {self.video_path}")
            self.running = False
            return

        while self.running:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # カラー変換を先に実行
            frame = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT), interpolation=cv2.INTER_LINEAR)  # 高速な補間方法を選択
            self.frame_ready.emit(frame)
            self.msleep(16)  # 60FPS相当

        cap.release()
        self.running = False
