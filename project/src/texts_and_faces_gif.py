import asyncio
import logging
import os
import time

import cv2
import numpy as np
import pyaudio
import boto3
from botocore.exceptions import ClientError
from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.handlers import TranscriptResultStreamHandler
from amazon_transcribe.model import TranscriptEvent
import screeninfo
from PIL import Image, ImageSequence



# ===============================
#   設定・定数の定義
# ===============================
SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2
CHANNEL_NUMS = 1
CHUNK_SIZE = 1024  # 1KB chunks
REGION = "ap-northeast-1"
TRANSCRIPT_LANGUAGE_CODE = "ja-JP"
MEDIA_ENCODING = "pcm"
COMPREHEND_LANGUAGE_CODE = "ja"
WINDOW_WIDTH = 3840
WINDOW_HEIGHT = int(WINDOW_WIDTH * 10 / 16)

# サイレント判定の秒数
SILENT_SECONDS = 3
SILENT_SECONDS2 = 5
SILENT_SECONDS3 = 7

# 顔検出
SCALE_FACTOR = 0.15

# 画像パス(相対パスを使う場合は実行時のCWDに注意)
IMAGE_BLACK = "./images/black.png"
IMAGE_TALKING = "./images/talking.png"
IMAGE_TALKING_POSITIVE = "./images/talking_positive.png"
IMAGE_TALKING_NEGATIVE = "./images/talking_negative.png"
IMAGE_COMIC1 = "./images/comic-effect1.png"
IMAGE_COMIC4 = "./images/comic-effect4.png"
IMAGE_THINK1 = "./images/thinking1.png"
IMAGE_THINK2 = "./images/thinking2.png"
IMAGE_THINK3 = "./images/thinking3.png"


# gifパス
HAPPY_ANIMATION = "./gifs/happy_animation.gif"
SURPRISED_ANIMATION = "./gifs/surprised_animation.gif"
TALKING_ANIMATION = "./gifs/talking_animation.gif"
# todo: taking positive, negativeも作る


# ===============================
#   ロガーのセットアップ
# ===============================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ===============================
#   システム全体の状態を持つクラス
# ===============================
class SystemState:
    """共有したい状態をまとめるクラス"""
    def __init__(self):
        # 音声が検出された最後のタイムスタンプを保持
        self.last_audio_time = time.time()


# ===============================
#   画像表示管理クラス
# ===============================
class ImageWindowManager:
    """OpenCVウィンドウや画像表示を管理するクラス"""
    def __init__(self, window_name: str = "Image"):
        self.window_name = window_name

        # モニター情報を取得
        monitors = screeninfo.get_monitors()
        num_monitors = len(monitors)
        logger.info(f"monitors: {monitors}")
        logger.info(f"Number of monitors: {num_monitors}")
        if num_monitors < 2:
            monitor = monitors[0]
        else:
            # 2番目のモニターを使うなど用途に応じて変更
            monitor = monitors[1]

        logger.info(f"monitor: {monitor}")
        self.screen_width = monitor.width
        self.screen_height = monitor.height
        self._setup_window(monitor)

    def _setup_window(self, monitor):
        """OpenCVウィンドウ初期化"""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        # 指定モニターにウィンドウを移動
        cv2.moveWindow(self.window_name, monitor.x, monitor.y)
        # 画面全体を白(または黒)で初期化
        cv2.imshow(self.window_name, 255 * np.ones(
            (self.screen_height, self.screen_width), dtype=np.uint8))
        # フルスクリーン化
        cv2.setWindowProperty(
            self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
        )

    def display_image(self, image_path: str):
        """画像を読み込み、16:10にリサイズして表示"""
        abs_path = os.path.abspath(image_path)
        if not os.path.exists(abs_path):
            logger.error(f"Error: The path {abs_path} does not exist.")
            return

        image = cv2.imread(abs_path)
        if image is None:
            logger.error(f"Error: Unable to load image at {abs_path}")
            return
        
        resized_image = cv2.resize(image, (WINDOW_WIDTH, WINDOW_HEIGHT))
        cv2.imshow(self.window_name, resized_image)

    def read_gif(self, gif_path: str):
        """GIFを読み込み、各フレームをRGB->BGRに変換して返す"""
        gif = Image.open(gif_path)
        frames = []
        for frame in ImageSequence.Iterator(gif):
            # RGB -> BGRに変換
            frame_array = np.array(frame.convert('RGB'))[..., ::-1]
            frames.append(frame_array)
        return np.array(frames)

    def show_gif(self, gif_frames, interval:int):
        """GIFの各フレームを表示"""
        for t in range(len(gif_frames)):
            resized_frame = cv2.resize(gif_frames[t], (WINDOW_WIDTH, WINDOW_HEIGHT))
            cv2.imshow(self.window_name, resized_frame)
            cv2.waitKey(interval)


    async def event_loop(self):
        """OpenCVのイベントループを非同期でまわす"""
        while True:
            # 1ms待機 & キーイベント処理
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("OpenCV window closed by user.")
                break
            await asyncio.sleep(0.01)

    def close(self):
        """ウィンドウを閉じる"""
        cv2.destroyAllWindows()
        logger.info("OpenCV windows closed.")


# ===============================
#   Comprehendを用いた感情分析
# ===============================
class ComprehendDetect:
    """Handles sentiment detection using Amazon Comprehend."""
    def __init__(self, window_manager: ImageWindowManager):
        self.window_manager = window_manager
        # Boto3クライアント生成
        self.comprehend_client = boto3.client('comprehend', region_name=REGION)

    def detect_sentiment(self, text: str, language_code: str):
        try:
            response = self.comprehend_client.detect_sentiment(
                Text=text,
                LanguageCode=language_code
            )
            sentiment = response["Sentiment"]
            logger.info("Detected sentiment: %s", sentiment)
            # 音声の感情分析結果によって画像を変更
            if sentiment == 'POSITIVE':
                self.window_manager.display_image(IMAGE_TALKING_POSITIVE)
            elif sentiment == 'NEGATIVE':
                self.window_manager.display_image(IMAGE_TALKING_NEGATIVE)
            else:
                talking_gif = self.window_manager.read_gif(TALKING_ANIMATION)
                self.window_manager.show_gif(talking_gif, 100)
            return sentiment
        except ClientError as error:
            logger.error("Error detecting sentiment: %s", error)
            raise


# ===============================
#   Transcribeイベントのハンドラ
# ===============================
class MyEventHandler(TranscriptResultStreamHandler):
    """Handles transcription events and performs sentiment analysis."""
    def __init__(self, output_stream, comp_detect: ComprehendDetect):
        super().__init__(output_stream)
        self.comp_detect = comp_detect

    async def handle_transcript_event(self, transcript_event: TranscriptEvent):
        """Processes transcription event and detects sentiment."""
        results = transcript_event.transcript.results
        for result in results:
            for alt in result.alternatives:
                transcript_text = alt.transcript
                logger.info("Transcription: %s", transcript_text)
                await self.analyze_sentiment(transcript_text)

    async def analyze_sentiment(self, text: str):
        """Performs sentiment analysis on transcribed text."""
        self.comp_detect.detect_sentiment(text, COMPREHEND_LANGUAGE_CODE)


# ===============================
#   音声入力とTranscribeの処理
# ===============================
async def basic_transcribe(state: SystemState, comp_detect: ComprehendDetect):
    """Captures audio from microphone and sends it to Amazon Transcribe."""
    client = TranscribeStreamingClient(region=REGION)

    # ストリーミング開始
    stream = await client.start_stream_transcription(
        language_code=TRANSCRIPT_LANGUAGE_CODE,
        media_sample_rate_hz=SAMPLE_RATE,
        media_encoding=MEDIA_ENCODING,
    )
    logger.info("Started transcription stream.")

    p = pyaudio.PyAudio()
    audio_stream = None

    try:
        audio_stream = p.open(format=pyaudio.paInt16,
                              channels=CHANNEL_NUMS,
                              rate=SAMPLE_RATE,
                              input=True,
                              frames_per_buffer=CHUNK_SIZE)
        logger.info("Audio stream opened.")

        async def write_chunks():
            """音声を取得してTranscribeに送るタスク"""
            try:
                while True:
                    data = await asyncio.to_thread(
                        audio_stream.read, CHUNK_SIZE, exception_on_overflow=False
                    )
                    audio_np = np.frombuffer(data, dtype=np.int16)
                    if np.abs(audio_np).mean() > 100:
                        state.last_audio_time = time.time()
                    await stream.input_stream.send_audio_event(audio_chunk=data)

            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.error(f"Error in write_chunks: {e}")
            finally:
                await stream.input_stream.end_stream()
                logger.info("Ended transcription stream.")

        # イベントハンドラをセットアップ
        handler = MyEventHandler(stream.output_stream, comp_detect)
        # 非同期で同時実行
        await asyncio.gather(write_chunks(), handler.handle_events())

    except ClientError as e:
        logger.error(f"ClientError in basic_transcribe: {e}")
    except Exception as e:
        logger.error(f"Unexpected error in basic_transcribe: {e}")
    finally:
        if audio_stream:
            audio_stream.stop_stream()
            audio_stream.close()
            logger.info("Audio stream closed.")
        p.terminate()
        logger.info("PyAudio terminated.")


# ===============================
#   顔認識と感情表示を行うクラス
# ===============================
class FaceMonitor:
    """Webカメラ映像を取得し、Rekognitionで感情を検出"""
    def __init__(self, window_manager: ImageWindowManager):
        self.window_manager = window_manager
        self.cap = cv2.VideoCapture(0)
        # Boto3 (Rekognition) クライアント
        session = boto3.Session(profile_name="rekognition")
        self.rekognition = session.client('rekognition')

    async def run(self, state: SystemState):
        """表情を定期的に検出し、画像を切り替える"""
        while True:
            current_time = time.time()

            # フレームを取得
            ret, frame = self.cap.read()
            if not ret:
                logger.error("Failed to read from camera.")
                await asyncio.sleep(1)
                continue

            height, width, channels = frame.shape
            # サイズ縮小
            small = cv2.resize(
                frame, (int(width * SCALE_FACTOR), int(height * SCALE_FACTOR))
            )
            ret, buf = cv2.imencode('.jpg', small)
            if not ret:
                logger.error("Failed to encode frame to .jpg.")
                await asyncio.sleep(1)
                continue

            # Rekognitionで感情分析
            faces = self.rekognition.detect_faces(
                Image={'Bytes': buf.tobytes()},
                Attributes=['ALL']
            )
            # 1秒以上音声入力がなかったら表情をチェック
            if current_time - state.last_audio_time > 1:
                # いったん黒画面に
                self.window_manager.display_image(IMAGE_BLACK)
                for face in faces.get('FaceDetails', []):
                    emotions = face.get('Emotions', [])
                    if not emotions:
                        continue
                    first_emotion = emotions[0]
                    emotion_type = first_emotion['Type']
                    logger.info("Detected emotion: %s", emotion_type)

                    if emotion_type == 'SURPRISED':
                        surprised_gif = self.window_manager.read_gif(SURPRISED_ANIMATION)
                        self.window_manager.show_gif(surprised_gif, 5)
                        state.last_audio_time = current_time
                    elif emotion_type == 'HAPPY':
                        happy_gif = self.window_manager.read_gif(HAPPY_ANIMATION)
                        self.window_manager.show_gif(happy_gif, 100)
                        state.last_audio_time = current_time
                    else:
                        # サイレント判定による画像切り替え
                        silence_duration = current_time - state.last_audio_time
                        if SILENT_SECONDS <= silence_duration < SILENT_SECONDS2:
                            self.window_manager.display_image(IMAGE_THINK1)
                        elif SILENT_SECONDS2 <= silence_duration < SILENT_SECONDS3:
                            self.window_manager.display_image(IMAGE_THINK2)
                        elif silence_duration >= SILENT_SECONDS3:
                            self.window_manager.display_image(IMAGE_THINK3)
                            state.last_audio_time = current_time

            await asyncio.sleep(1)

    def release(self):
        """カメラリソースを解放"""
        if self.cap.isOpened():
            self.cap.release()
            logger.info("VideoCapture released.")


# ===============================
#   メインエントリーポイント
# ===============================
async def main():
    logger.info("Setting up system...")

    # システム状態とウィンドウ管理を初期化
    state = SystemState()
    window_manager = ImageWindowManager()
    comp_detect = ComprehendDetect(window_manager)

    # 初期画面を表示
    window_manager.display_image(IMAGE_BLACK)

    # GIFの読み込み（window_manager の初期化後）
    # happy_gif = window_manager.read_gif(HAPPY_ANIMATION)
    # surprised_gif = window_manager.read_gif(SURPRISED_ANIMATION)

    # 顔認識モニターを準備
    face_monitor = FaceMonitor(window_manager)

    # 非同期タスクを並行で動かす
    # 1) 音声入力 & Transcribe
    # 2) カメラ映像で表情検出
    # 3) OpenCVのイベントループ
    await asyncio.gather(
        basic_transcribe(state, comp_detect),
        face_monitor.run(state),
        window_manager.event_loop()
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Stopped by user.")
    except Exception as e:
        logger.error(f"Unexpected error in main: {e}")
    finally:
        cv2.destroyAllWindows()
        logger.info("OpenCV windows closed.")
