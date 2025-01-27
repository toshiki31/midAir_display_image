import sys
import asyncio
import logging
import time
import cv2
import boto3
import numpy as np
import pyaudio
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QThread, QTimer, pyqtSignal
from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.handlers import TranscriptResultStreamHandler
from amazon_transcribe.model import TranscriptEvent
from PIL import Image, ImageSequence

# ===============================
#   設定・定数の定義
# ===============================
SAMPLE_RATE = 16000
CHUNK_SIZE = 1024
TRANSCRIPT_LANGUAGE_CODE = "ja-JP"
MEDIA_ENCODING = "pcm"
COMPREHEND_LANGUAGE_CODE = "ja"
WINDOW_WIDTH = 2500
WINDOW_HEIGHT = int(WINDOW_WIDTH * 10 / 16)

# パス設定
IMAGE_BLACK = "./images/black.png"
IMAGE_THINKING1 = "./images/thinking1.png"
IMAGE_THINKING2 = "./images/thinking2.png"
IMAGE_THINKING3 = "./images/thinking3.png"
HAPPY_MOVIE = "./movies/happy_movie.mp4"
SURPRISED_MOVIE = "./movies/surprised_movie.mp4"
TALKING_MOVIE = "./movies/talking_movie.mp4"
TALKING_POSITIVE_MOVIE = "./movies/talking_positive_movie.mp4"
TALKING_NEGATIVE_MOVIE = "./movies/talking_negative_movie.mp4"

# ロガーセットアップ
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Rekognition & Comprehend クライアント
rekognition = boto3.client("rekognition", region_name="ap-northeast-1")
comprehend = boto3.client("comprehend", region_name="ap-northeast-1")


# ===============================
#   ウィンドウ管理クラス
# ===============================
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


# ===============================
#   Comprehendで感情分析
# ===============================
class ComprehendDetect:
    """Comprehendで感情分析を行い、動画を再生"""
    def __init__(self, window: FullScreenVideoWindow, video_thread: VideoThread):
        self.window = window
        self.video_thread = video_thread

    def detect_sentiment(self, text):
        try:
            response = comprehend.detect_sentiment(Text=text, LanguageCode=COMPREHEND_LANGUAGE_CODE)
            sentiment = response["Sentiment"]
            logger.info(f"Detected sentiment: {sentiment}")

            # 感情に応じた動画を再生
            if sentiment == "POSITIVE":
                if not self.video_thread.isRunning():
                    self.video_thread.start_video(TALKING_POSITIVE_MOVIE)
            elif sentiment == "NEGATIVE":
                if not self.video_thread.isRunning():
                    self.video_thread.start_video(TALKING_NEGATIVE_MOVIE)
            else:
                if not self.video_thread.isRunning():
                    self.video_thread.start_video(TALKING_MOVIE)
        except Exception as e:
            logger.error(f"Error detecting sentiment: {e}")


# ===============================
#   音声認識スレッド
# ===============================
class MyEventHandler(TranscriptResultStreamHandler):
    def __init__(self, output_stream, comprehend_detect):
        super().__init__(output_stream)
        self.comprehend_detect = comprehend_detect

    async def handle_transcript_event(self, transcript_event: TranscriptEvent):
        results = transcript_event.transcript.results
        for result in results:
            for alt in result.alternatives:
                transcript = alt.transcript
                logger.info(f"Transcription: {transcript}")
                self.comprehend_detect.detect_sentiment(transcript)


# ===============================
#   音声認識スレッド (修正済み)
# ===============================
class AudioTranscriptionThread(QThread):
    """音声認識スレッド"""
    audio_detected = pyqtSignal()  # 音声が検出されたシグナル

    def __init__(self, comprehend_detect):
        super().__init__()
        self.comprehend_detect = comprehend_detect

    def run(self):
        asyncio.run(self.transcribe_stream())

    async def transcribe_stream(self):
        client = TranscribeStreamingClient(region="ap-northeast-1")
        stream = await client.start_stream_transcription(
            language_code=TRANSCRIPT_LANGUAGE_CODE,
            media_sample_rate_hz=SAMPLE_RATE,
            media_encoding=MEDIA_ENCODING,
        )
        logger.info("Started transcription stream.")

        p = pyaudio.PyAudio()
        audio_stream = p.open(format=pyaudio.paInt16,
                              channels=1,
                              rate=SAMPLE_RATE,
                              input=True,
                              frames_per_buffer=CHUNK_SIZE)
        logger.info("Audio stream opened.")

        async def write_chunks():
            try:
                while True:
                    data = await asyncio.to_thread(
                        audio_stream.read, CHUNK_SIZE, exception_on_overflow=False
                    )
                    audio_np = np.frombuffer(data, dtype=np.int16)
                    if np.abs(audio_np).mean() > 100:
                        self.audio_detected.emit()
                    await stream.input_stream.send_audio_event(audio_chunk=data)
            except asyncio.CancelledError:
                pass
            finally:
                await stream.input_stream.end_stream()

        handler = MyEventHandler(stream.output_stream, self.comprehend_detect)
        await asyncio.gather(write_chunks(), handler.handle_events())

        audio_stream.stop_stream()
        audio_stream.close()
        p.terminate()


# ===============================
#   メインアプリケーション (修正済み)
# ===============================
class EmotionApp(QWidget):
    """メインアプリケーション"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Emotion Analysis")
        self.setGeometry(100, 100, 800, 600)
        self.last_audio_time = time.time()  # 初期化
        self.SILENT_SECONDS = 3
        self.SILENT_SECONDS2 = 5
        self.SILENT_SECONDS3 = 7
        self.SILENT_SECONDS4 = 9

        self.camera_label = QLabel(self)
        self.camera_label.setFixedSize(640, 480)
        layout = QVBoxLayout()
        layout.addWidget(self.camera_label)
        self.setLayout(layout)

        # Rekognition用のカメラ設定
        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_camera)

        # 動画再生用ウィンドウ
        self.video_window = FullScreenVideoWindow()
        self.video_window.move_to_secondary_screen()

        # 動画スレッド
        self.video_thread = VideoThread()
        self.video_thread.frame_ready.connect(self.video_window.frame_ready)
        self.video_thread.finished.connect(self.clear_video)

        # Comprehend検出器
        self.comprehend_detect = ComprehendDetect(self.video_window, self.video_thread)

        # 音声認識スレッド
        self.audio_thread = AudioTranscriptionThread(self.comprehend_detect)
        self.audio_thread.audio_detected.connect(self.reset_audio_timer)
        self.audio_thread.start()

        # 音声検出用タイマー
        self.audio_timer = QTimer(self)
        self.audio_timer.timeout.connect(self.enable_camera_detection)
        self.audio_timer.setInterval(1000)  # 1秒
        self.audio_timer.start()

        # 表情認識のタイマー状態
        self.is_camera_active = False

        # 起動時に黒画面を初期表示
        self.display_image(IMAGE_BLACK)

    def reset_audio_timer(self):
        """音声が検出されたときにタイマーをリセット"""
        self.audio_timer.start()
        if self.is_camera_active:
            self.timer.stop()
            self.is_camera_active = False

    def enable_camera_detection(self):
        """音声が1秒間検出されない場合に表情認識を有効化"""
        if not self.is_camera_active:
            self.timer.start(30)
            self.is_camera_active = True

    def update_camera(self):
        """表情認識の処理"""
        ret, frame = self.cap.read()
        if not ret:
            return

        current_time = time.time()
        silence_duration = current_time - self.last_audio_time

        # カメラの画像を取得して顔認識を実行
        small_frame = cv2.resize(frame, (320, 240))
        _, buf = cv2.imencode('.jpg', small_frame)
        response = rekognition.detect_faces(Image={'Bytes': buf.tobytes()}, Attributes=['ALL'])

        # カメラ画像の更新
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        qimg = QImage(frame.data, w, h, ch * w, QImage.Format_RGB888)
        self.camera_label.setPixmap(QPixmap.fromImage(qimg))

        # Rekognition で顔が検出されない場合の処理
        if not response.get('FaceDetails'):
            logger.info("No faces detected, keeping the current state.")
            return

        for face in response.get('FaceDetails', []):
            emotions = face.get('Emotions', [])
            if not emotions:
                # 表情が検出されない場合はログに記録し、直前の状態を維持
                logger.info("No emotions detected, keeping the current state.")
                return

            first_emotion = emotions[0]
            emotion_type = first_emotion['Type']
            logger.info(f"Detected emotion: {emotion_type}")

            if emotion_type == 'HAPPY' and not self.video_thread.isRunning():
                self.video_thread.start_video(HAPPY_MOVIE)
                self.last_audio_time = current_time
            elif emotion_type == 'SURPRISED' and not self.video_thread.isRunning():
                self.video_thread.start_video(SURPRISED_MOVIE)
                self.last_audio_time = current_time
            else:
                # 喋っていない秒数に応じて THINKING 画像を切り替え
                if self.SILENT_SECONDS <= silence_duration < self.SILENT_SECONDS2:
                    self.display_image(IMAGE_THINKING1)
                elif self.SILENT_SECONDS2 <= silence_duration < self.SILENT_SECONDS3:
                    self.display_image(IMAGE_THINKING2)
                elif self.SILENT_SECONDS3 <= silence_duration < self.SILENT_SECONDS4:
                    self.display_image(IMAGE_THINKING3)
                elif  silence_duration >= self.SILENT_SECONDS4:
                    self.display_image(IMAGE_BLACK)  # 黒画面を表示
                    self.last_audio_time = current_time

    
    def display_image(self, image_path: str):
        """画像を読み込み、16:10にリサイズして表示"""
        import os

        abs_path = os.path.abspath(image_path)
        if not os.path.exists(abs_path):
            logger.error(f"Error: The path {abs_path} does not exist.")
            return

        image = cv2.imread(abs_path)
        if image is None:
            logger.error(f"Error: Unable to load image at {abs_path}")
            return

        resized_image = cv2.resize(image, (WINDOW_WIDTH, WINDOW_HEIGHT))
        
        # PyQt ウィジェットに表示
        h, w, ch = resized_image.shape
        qimg = QImage(resized_image.data, w, h, ch * w, QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(qimg)
        self.video_window.video_label.setPixmap(pixmap)

    def clear_video(self):
        """動画再生終了後に表示をクリア"""
        self.video_window.video_label.clear()

    def closeEvent(self, event):
        """アプリケーション終了時の処理"""
        self.cap.release()
        self.video_thread.stop()
        event.accept()



if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = EmotionApp()
    main_window.show()
    sys.exit(app.exec_())
