import sys
import asyncio
import logging
import time
import cv2
import boto3
import numpy as np
import pyaudio
from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.handlers import TranscriptResultStreamHandler
from amazon_transcribe.model import TranscriptEvent
from PIL import Image, ImageTk
import tkinter as tk
import screeninfo
import threading

# ===============================
#   定数・設定の定義
# ===============================
SAMPLE_RATE = 16000
CHUNK_SIZE = 1024
TRANSCRIPT_LANGUAGE_CODE = "ja-JP"
MEDIA_ENCODING = "pcm"
COMPREHEND_LANGUAGE_CODE = "ja"
WINDOW_WIDTH = 2500
WINDOW_HEIGHT = int(WINDOW_WIDTH * 10 / 16)

# パス設定（必要に応じて使用）
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

# Rekognition & Comprehend クライアント（本コードでは直接使用していませんが、必要に応じて削除可）
rekognition = boto3.client("rekognition", region_name="ap-northeast-1")
comprehend = boto3.client("comprehend", region_name="ap-northeast-1")


# ===============================
#   音声認識スレッド (Tkinter版)
# ===============================
class AudioTranscriptionThread(threading.Thread):
    """音声認識スレッド（Tkinter用：PyQt部分を除去）"""
    def __init__(self, speech_bubble):
        super().__init__()
        self.speech_bubble = speech_bubble
        self.loop = None  # asyncio のループを保持

    def run(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self.transcribe_stream())

    async def transcribe_stream(self):
        try:
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
                        try:
                            data = await asyncio.to_thread(
                                audio_stream.read, CHUNK_SIZE, exception_on_overflow=False
                            )
                        except IOError as e:
                            logger.error(f"Audio read error: {e}")
                            continue
                        audio_np = np.frombuffer(data, dtype=np.int16)
                        # しきい値以上の音量ならログ出力（ここではシグナルの代わり）
                        if np.abs(audio_np).mean() > 100:
                            logger.info("Audio detected!")
                        await stream.input_stream.send_audio_event(audio_chunk=data)
                except asyncio.CancelledError:
                    pass
                finally:
                    await stream.input_stream.end_stream()

            # MyEventHandler にメインスレッドで生成した SpeechBubble を渡す
            handler = MyEventHandler(stream.output_stream, self.speech_bubble)
            await asyncio.gather(write_chunks(), handler.handle_events())

        except Exception as e:
            logger.error(f"Error in transcribe_stream: {e}")

        finally:
            audio_stream.stop_stream()
            audio_stream.close()
            p.terminate()


# ===============================
#   音声認識 -> 吹き出し表示
# ===============================
class MyEventHandler(TranscriptResultStreamHandler):
    def __init__(self, output_stream, speech_bubble):
        super().__init__(output_stream)
        self.speech_bubble = speech_bubble

    async def handle_transcript_event(self, transcript_event: TranscriptEvent):
        results = transcript_event.transcript.results
        for result in results:
            for alt in result.alternatives:
                transcript = alt.transcript
                logger.info(f"Transcription: {transcript}")  # 文字起こし結果をログ出力
                # メインスレッドで SpeechBubble の update_text を実行する
                self.speech_bubble.root.after(0, self.speech_bubble.update_text, transcript)


# ===============================
#   吹き出し表示クラス (SpeechBubble)
# ===============================
class SpeechBubble:
    def __init__(self):
        """
        吹き出し表示用の別ウィンドウを作成し、背景画像とテキスト表示領域を初期化する。
        このインスタンスは必ずメインスレッドで生成してください。
        """
        # メインウィンドウは非表示にしておく
        self.root = tk.Tk()
        self.root.withdraw()  # メインウィンドウを隠す

        # 吹き出しウィンドウ（ポップアップ）の作成
        self.popup = tk.Toplevel(self.root)
        self.popup.title("Speech Bubble")
        self.popup.attributes("-topmost", True)  # 常に最前面に表示
        self.popup.update_idletasks()
        self.popup.update()

        # スクリーン情報の取得（複数モニター対応）
        monitors = screeninfo.get_monitors()
        if len(monitors) > 1:
            second_monitor = monitors[1]
            monitor_width = second_monitor.width
            monitor_height = second_monitor.height
            monitor_x = second_monitor.x
            monitor_y = second_monitor.y
        else:
            monitor_width = self.root.winfo_screenwidth()
            monitor_height = self.root.winfo_screenheight()
            monitor_x = 0
            monitor_y = 0

        # 16:10 のアスペクト比を考慮したウィンドウサイズの計算
        aspect_ratio = 16 / 10
        popup_width = int(monitor_height * aspect_ratio)
        popup_height = monitor_height
        self.popup.geometry(f"{popup_width}x{popup_height}+{monitor_x}+{monitor_y}")

        # 背景画像（吹き出し画像）の読み込みとリサイズ
        self.bg_image = Image.open("images/fukidashi.png")
        self.bg_image = self.bg_image.resize((popup_width, 200), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(self.bg_image)

        # Canvas を作成して背景画像を表示
        self.canvas = tk.Canvas(self.popup, width=popup_width, height=popup_height)
        self.canvas.pack()
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        # Canvas にテキストを表示するための領域を作成
        self.text_id = self.canvas.create_text(popup_width / 2, 80,
                                                text="",
                                                font=("Arial", 100, "bold"),
                                                fill="black")
        
    def update_text(self, text):
        """
        吹き出し内に表示するテキストを更新する。
        """
        self.canvas.itemconfig(self.text_id, text=text)
        self.root.update_idletasks()

    def show(self):
        """
        吹き出しウィンドウを表示する。
        """
        self.popup.deiconify()
        self.root.update()

    def hide(self):
        """
        吹き出しウィンドウを非表示にする。
        """
        self.popup.withdraw()
        self.root.update()


# ===============================
#   メインアプリケーション (Tkinter版)
# ===============================
def main():
    # SpeechBubble はメインスレッドで生成
    speech_bubble = SpeechBubble()
    speech_bubble.show()

    # 音声認識スレッドを開始
    audio_thread = AudioTranscriptionThread(speech_bubble)
    audio_thread.daemon = True
    audio_thread.start()

    # Tkinter のメインループを開始
    speech_bubble.root.mainloop()


if __name__ == "__main__":
    main()
