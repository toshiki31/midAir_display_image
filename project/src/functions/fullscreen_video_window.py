from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout, QApplication
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import pyqtSignal
import numpy as np

class FullScreenVideoWindow(QWidget):
    """第2モニターにフルスクリーンで表示するウィンドウ"""
    frame_ready = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Full Screen Video Window")
        self.video_label = QLabel(self)
        self.video_label.setStyleSheet("background-color: black;")  # 背景色を黒に設定
        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        self.setLayout(layout)

        # シグナルでフレームを更新
        self.frame_ready.connect(self.update_frame)

    def update_frame(self, frame):
        """動画フレームを更新"""
        h, w, ch = frame.shape
        qimg = QImage(frame.data, w, h, ch * w, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qimg))

    def move_to_secondary_screen(self):
        """第2モニターに移動してフルスクリーン表示"""
        desktop = QApplication.desktop()
        screen_count = desktop.screenCount()

        if screen_count > 1:
            # 第2モニターのジオメトリを取得
            screen_geometry = desktop.screenGeometry(1)  # 第2モニター
        else:
            # 第1モニターにフォールバック
            screen_geometry = desktop.screenGeometry(0)

        self.setGeometry(screen_geometry)
        self.showFullScreen()
