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
import tkinter.font as tkfont
import screeninfo
import threading

# ===============================
#   定数・設定の定義
# ===============================
SAMPLE_RATE = 16000
CHUNK_SIZE = 1024
MEDIA_ENCODING = "pcm"
COMPREHEND_LANGUAGE_CODE = "ja"
WINDOW_WIDTH = 2500
WINDOW_HEIGHT = int(WINDOW_WIDTH * 10 / 16)
TRANSCRIPT_LANGUAGE_CODE = "ja-JP" # 初期値
TARGET_LANGUAGE_CODE = "en-US"  # 初期値
REGION = "ap-northeast-1"
SPEECH_BUBBLE1 = "./images/speech-bubble1.png"
SPEECH_BUBBLE2 = "./images/speech-bubble2.png"
SPEECH_BUBBLE3 = "./images/speech-bubble3.png"

# ロガーセットアップ
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ===============================
#   言語選択ダイアログ
# ===============================
def get_language_settings():
    lang_root = tk.Tk()
    lang_root.withdraw()
    dialog = tk.Toplevel(lang_root)
    dialog.title("言語設定")
    tk.Label(dialog, text="翻訳元言語:").grid(row=0, column=0, padx=10, pady=10)
    tk.Label(dialog, text="翻訳先言語:").grid(row=1, column=0, padx=10, pady=10)
    
    languages = ["ja-JP", "en-US", "fr-FR", "de-DE", "es-ES"]
    source_var = tk.StringVar(value=languages[0])
    target_var = tk.StringVar(value=languages[1])
    tk.OptionMenu(dialog, source_var, *languages).grid(row=0, column=1, padx=10, pady=10)
    tk.OptionMenu(dialog, target_var, *languages).grid(row=1, column=1, padx=10, pady=10)
    
    result = {}
    def on_ok():
        result['source'] = source_var.get()
        result['target'] = target_var.get()
        dialog.destroy()
    tk.Button(dialog, text="OK", command=on_ok).grid(row=2, column=0, columnspan=2, pady=10)
    dialog.grab_set()
    lang_root.wait_window(dialog)
    lang_root.destroy()
    return result['source'], result['target']


# ===============================
#   音声認識スレッド (Tkinter版)
# ===============================
class AudioTranscriptionThread(threading.Thread):
    """音声認識スレッド（Tkinter用）"""
    def __init__(self, speech_bubble):
        super().__init__()
        self.speech_bubble = speech_bubble
        self.loop = None

    def run(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self.transcribe_stream())

    async def transcribe_stream(self):
        try:
            client = TranscribeStreamingClient(region=REGION)
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
                        if np.abs(audio_np).mean() > 100:
                            logger.info("Audio detected!")
                        await stream.input_stream.send_audio_event(audio_chunk=data)
                except asyncio.CancelledError:
                    pass
                finally:
                    await stream.input_stream.end_stream()

            handler = MyEventHandler(stream.output_stream, self.speech_bubble)
            await asyncio.gather(write_chunks(), handler.handle_events())

        except Exception as e:
            logger.error(f"Error in transcribe_stream: {e}")
        finally:
            audio_stream.stop_stream()
            audio_stream.close()
            p.terminate()


# ===============================
#   音声認識 -> 翻訳 -> 吹き出し表示
# ===============================
class MyEventHandler(TranscriptResultStreamHandler):
    def __init__(self, output_stream, speech_bubble):
        super().__init__(output_stream)
        self.speech_bubble = speech_bubble

    def translate_text(self, text):
        translate = boto3.client("translate", region_name=REGION, use_ssl=True)
        result = translate.translate_text(
            Text=text,
            SourceLanguageCode=TRANSCRIPT_LANGUAGE_CODE,
            TargetLanguageCode=TARGET_LANGUAGE_CODE
        )
        return result.get("TranslatedText", text)

    async def handle_transcript_event(self, transcript_event: TranscriptEvent):
        results = transcript_event.transcript.results
        for result in results:
            for alt in result.alternatives:
                transcript = alt.transcript
                logger.info(f"Transcription: {transcript}")
                # 翻訳を非同期で実行
                translated = await asyncio.to_thread(self.translate_text, transcript)
                logger.info(f"Translated: {translated}")
                # 受信時刻を更新し、翻訳結果を表示
                self.speech_bubble.root.after(0, self.speech_bubble.update_text, translated)


# ===============================
#   吹き出し表示クラス (SpeechBubble)
# ===============================
class SpeechBubble:
    def __init__(self):
        """
        吹き出し表示用の別ウィンドウを作成し、背景画像とテキスト表示領域を初期化する。
        背景画像はウィンドウ全体の横幅に合わせ、元画像のアスペクト比を維持して自動で高さを決定する。
        このインスタンスは必ずメインスレッドで生成してください。
        """
        # メインウィンドウ（非表示）を生成
        self.root = tk.Tk()
        self.root.withdraw()

       # ポップアップウィンドウの生成
        self.popup = tk.Toplevel(self.root)
        self.popup.title("Speech Bubble")
        self.popup.attributes("-topmost", True)
        self.popup.update_idletasks()
        self.popup.update()

        # スクリーン情報の取得（複数モニターがある場合は2番目のモニターを使用）
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

        # ウィンドウの横幅を、モニターの高さとアスペクト比16:10から計算
        aspect_ratio = 16 / 10
        popup_width = int(monitor_height * aspect_ratio)
        popup_height = monitor_height
        self.popup.geometry(f"{popup_width}x{popup_height}+{monitor_x}+{monitor_y}")

        # Canvas を作成（背景色はグレー）
        self.canvas = tk.Canvas(self.popup, width=popup_width, height=popup_height, bg="lightgray")
        self.canvas.pack()

        # 吹き出し画像の表示処理は削除（画像は表示せず、背景はグレー）

        # テキスト表示領域の設定（余白を取る）
        self.text_width_limit = popup_width - 100
        self.text_id = self.canvas.create_text(popup_width / 2, popup_height // 2,
                                                text="",
                                                font=("Arial", 100, "bold"),
                                                fill="white",
                                                width=self.text_width_limit)
        self.text_font = tkfont.Font(family="Arial", size=100, weight="bold")
        # 初回の発話時刻の初期化
        self.last_transcript_time = time.time()
        self.poll_silence()

    def poll_silence(self):
        """
        発話が一定時間ない場合は、字幕を非表示にする。
        """
        if time.time() - self.last_transcript_time >= 1:
            self.canvas.itemconfig(self.text_id, state='hidden')
        else:
            self.canvas.itemconfig(self.text_id, state='normal')
        self.root.after(500, self.poll_silence)

    def update_text(self, text):
        """
        吹き出し内に表示するテキストを更新する。
        発話があった時刻を更新し、非表示状態を解除する。
        テキストが指定幅を超える場合は、右側（最新部分）のみ表示する。
        """
        self.last_transcript_time = time.time()
        self.canvas.itemconfig(self.text_id, state='normal')

        if self.text_font.measure(text) <= self.text_width_limit:
            display_text = text
        else:
            display_text = text
            for i in range(len(text)):
                substring = text[i:]
                if self.text_font.measure(substring) <= self.text_width_limit:
                    display_text = substring
                    break
        self.canvas.itemconfig(self.text_id, text=display_text)
        self.root.update_idletasks()

    def show(self):
        self.popup.deiconify()
        self.root.update()

    def hide(self):
        self.popup.withdraw()
        self.root.update()




# ===============================
#   メインアプリケーション (Tkinter版)
# ===============================
def main():
    source_lang, target_lang = get_language_settings()
    global TRANSCRIPT_LANGUAGE_CODE, TARGET_LANGUAGE_CODE
    TRANSCRIPT_LANGUAGE_CODE = source_lang
    TARGET_LANGUAGE_CODE = target_lang
    logger.info(f"Selected source language: {source_lang}, target language: {target_lang}")

    speech_bubble = SpeechBubble()
    speech_bubble.show()

    audio_thread = AudioTranscriptionThread(speech_bubble)
    audio_thread.daemon = True
    audio_thread.start()

    speech_bubble.root.mainloop()


if __name__ == "__main__":
    main()
