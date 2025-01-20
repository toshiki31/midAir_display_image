import sys
import cv2
import boto3
import numpy as np
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, QThread, pyqtSignal

# Rekognition Setup
rekognition = boto3.client('rekognition')
WINDOW_WIDTH = 2400 # 適宜変える
WINDOW_HEIGHT = int(WINDOW_WIDTH * 10 / 16)

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


class VideoApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Emotion-Based Video Display")
        self.setGeometry(100, 100, 800, 600)

        # カメラ表示用ラベル
        self.camera_label = QLabel(self)
        self.camera_label.setFixedSize(640, 480)

        # レイアウト設定
        layout = QVBoxLayout()
        layout.addWidget(self.camera_label)
        self.setLayout(layout)

        # カメラ設定
        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_camera)
        self.timer.start(30)  # 30msごとにフレームを更新

        # 動画再生用ウィンドウ
        self.video_window = FullScreenVideoWindow()
        self.video_window.move_to_secondary_screen()

        # 動画再生スレッド
        self.video_thread = VideoThread()
        self.video_thread.frame_ready.connect(self.video_window.frame_ready)
        self.video_thread.finished.connect(self.clear_video)

    def update_camera(self):
        """カメラフレームを更新"""
        ret, frame = self.cap.read()
        if not ret:
            return

        # フレームをRekognition用にリサイズ
        small_frame = cv2.resize(frame, (320, 240))
        _, buf = cv2.imencode('.jpg', small_frame)
        response = rekognition.detect_faces(Image={'Bytes': buf.tobytes()}, Attributes=['ALL'])

        # カメラフレーム表示
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        qimg = QImage(frame.data, w, h, ch * w, QImage.Format_RGB888)
        self.camera_label.setPixmap(QPixmap.fromImage(qimg))

        # 感情に応じた動画再生
        for face in response.get('FaceDetails', []):
            emotions = face['Emotions']
            if emotions:
                primary_emotion = emotions[0]['Type']
                if primary_emotion == 'HAPPY' and not self.video_thread.isRunning():
                    self.video_thread.start_video("./movies/happy_movie.mp4")
                elif primary_emotion == 'SURPRISED' and not self.video_thread.isRunning():
                    self.video_thread.start_video("./movies/surprised_movie.mp4")

    def clear_video(self):
        """動画再生終了後に表示をクリア"""
        self.video_window.video_label.clear()

    def closeEvent(self, event):
        """アプリケーション終了時の処理"""
        self.cap.release()
        self.video_thread.stop()
        event.accept()


class VideoThread(QThread):
    frame_ready = pyqtSignal(np.ndarray)
    finished = pyqtSignal()

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
        """動画再生スレッド"""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video {self.video_path}")
            self.running = False
            self.finished.emit()
            return

        while self.running:
            ret, frame = cap.read()
            if not ret:
                break

            # フレームのリサイズと変換
            frame = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.frame_ready.emit(frame)
            self.msleep(16)  # 約16fpsで再生

        cap.release()
        self.running = False
        self.finished.emit()

    def stop(self):
        """動画再生を停止"""
        self.running = False
        self.wait()


# メイン関数
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoApp()
    window.show()
    sys.exit(app.exec_())
