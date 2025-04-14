# .tobiienv内で実行してください

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
import tobii_research as tr
import csv
from datetime import datetime

# ===============================
#   定数・設定の定義
# ===============================
SAMPLE_RATE = 16000
CHUNK_SIZE = 1024
MEDIA_ENCODING = "pcm"
COMPREHEND_LANGUAGE_CODE = "ja"
WINDOW_WIDTH = 2500
WINDOW_HEIGHT = int(WINDOW_WIDTH * 10 / 16)
TRANSCRIPT_LANGUAGE_CODE = "ja-JP"  # 初期値
TARGET_LANGUAGE_CODE = "en-US"    # 初期値
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

        # 保存用にウィンドウサイズ・スクリーン情報を保持
        self.popup_width = popup_width
        self.popup_height = popup_height
        self.monitor_height = monitor_height

        # 背景画像の読み込み
        bg_image_path = SPEECH_BUBBLE1
        self.bg_image_orig = Image.open(bg_image_path)
        # 元画像サイズを取得
        orig_width, orig_height = self.bg_image_orig.size
        # ウィンドウ横幅に合わせた新しいサイズを計算（高さはアスペクト比に従う）
        new_width = popup_width
        new_height = int(orig_height * new_width / orig_width)
        self.bg_image = self.bg_image_orig.resize((new_width, new_height), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(self.bg_image)
        self.bg_height = new_height

        # Canvas を作成し、背景画像を配置
        self.canvas = tk.Canvas(self.popup, width=popup_width, height=popup_height, bg="black")
        self.canvas.pack()
        # 初期位置は y=0 として背景画像を配置
        self.bg_image_id = self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        # 字幕（テキスト）は背景画像の中央に配置（初期位置）
        self.text_id = self.canvas.create_text(popup_width / 2, new_height // 2,
                                                text="",
                                                font=("Arial", 200, "bold"),
                                                fill="black",
                                                width=popup_width - 100)
        self.text_font = tkfont.Font(family="Arial", size=200, weight="bold")

        self.last_transcript_time = time.time()
        self.poll_silence()

        # キーボードイベントのバインド
        self.popup.bind("<Key>", self.on_key_press)
        self.popup.focus_set()
        # ウィンドウの閉じるボタンが押されたときの処理
        self.popup.protocol("WM_DELETE_WINDOW", self.on_close)

    def update_bg_position(self, new_y):
        """
        キャンバス上の背景画像と字幕の y 座標を更新する。
        背景画像は new_y に、字幕は背景画像の高さの半分だけ下げた位置に設定する。
        """
        self.canvas.coords(self.bg_image_id, 0, new_y)
        text_y = new_y + self.bg_height // 2
        self.canvas.coords(self.text_id, self.popup_width / 2, text_y)

    def poll_silence(self):
        """
        発話が2秒以上ない場合は、背景画像と字幕を非表示にする。
        発話がある場合は表示状態を "normal" に戻す。
        """
        if time.time() - self.last_transcript_time >= 1:
            self.canvas.itemconfig(self.bg_image_id, state='hidden')
            self.canvas.itemconfig(self.text_id, state='hidden')
        else:
            self.canvas.itemconfig(self.bg_image_id, state='normal')
            self.canvas.itemconfig(self.text_id, state='normal')
        self.root.after(500, self.poll_silence)

    def update_text(self, text):
        """
        吹き出し内に表示するテキストを更新する。
        ・発話があった時刻を更新し、非表示状態を解除する。
        ・テキスト全体の幅が指定幅を超える場合は、右側（最新部分）のみ表示する。
        """
        self.last_transcript_time = time.time()
        self.canvas.itemconfig(self.bg_image_id, state='normal')
        self.canvas.itemconfig(self.text_id, state='normal')

        if self.text_font.measure(text) <= (self.popup_width - 100):
            display_text = text
        else:
            display_text = text
            for i in range(len(text)):
                substring = text[i:]
                if self.text_font.measure(substring) <= (self.popup_width - 100):
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
    
    def on_key_press(self, event):
        if event.char == 'q':
            logger.info("qキーが押されました。アプリケーションを終了します。")
            self.root.quit()  # mainloopを終了させる

    def on_close(self):
        logger.info("ウィンドウが閉じられました。Tobiiデータを保存します。")
        if tobii_thread and tobii_thread.running:
            tobii_thread.stop_streaming_and_save()
        self.root.quit()

# ===============================
#   顔検出スレッド (FaceDetectionThread)
# ===============================
class FaceDetectionThread(threading.Thread):
    """
    カメラ映像から顔検出を行い、検出された顔の上端に基づいて
    吹き出しウィンドウの背景画像と字幕の y 座標を更新するスレッドです。
    """
    def __init__(self, speech_bubble, offset=600):
        super().__init__()
        self.speech_bubble = speech_bubble
        self.offset = offset
        self.running = True
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def run(self):
        cap = cv2.VideoCapture(0)
        # memo: カメラ解像度の設定：popup の width と height と同じに設定する
        cam_width = self.speech_bubble.popup_width
        cam_height = self.speech_bubble.popup_height
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_height)
        while self.running:
            ret, frame = cap.read()
            if not ret:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            if len(faces) > 0:
                (x, y, w, h) = faces[0]
                face_top = y
                # todo: いい感じに位置調整する
                new_y = face_top - self.offset
                logger.info(f"new_y: {new_y}")
                if new_y < 0:
                    new_y = 0
                # 背景画像と字幕の両方の位置を更新する
                self.speech_bubble.root.after(0, self.speech_bubble.update_bg_position, new_y)
            time.sleep(0.1)
        cap.release()


class TobiiTrackingThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self.gaze_data_list = []
        self.running = False

        found_eyetrackers = tr.find_all_eyetrackers()
        if not found_eyetrackers:
            raise RuntimeError("アイトラッカーが見つかりません")
        self.my_eyetracker = found_eyetrackers[0]

    def gaze_data_callback(self, gaze_data):
        self.gaze_data_list.append(gaze_data)

    def start_streaming(self):
        if self.running:
            return
        self.running = True
        print("視線トラッキングを開始します")
        self.my_eyetracker.subscribe_to(tr.EYETRACKER_GAZE_DATA, self.gaze_data_callback, as_dictionary=True)

    def stop_streaming_and_save(self):
        if self.running:
            print("視線トラッキングを停止し、CSVに保存します")
            self.my_eyetracker.unsubscribe_from(tr.EYETRACKER_GAZE_DATA, self.gaze_data_callback)
            self.save_to_csv()
            self.running = False

    def save_to_csv(self):
        date = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        filename = f"{date}_gaze_data.csv"
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = ['left_eye_x', 'left_eye_y', 'right_eye_x', 'right_eye_y']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for data in self.gaze_data_list:
                writer.writerow({
                    'left_eye_x': data['left_gaze_point_on_display_area'][0],
                    'left_eye_y': data['left_gaze_point_on_display_area'][1],
                    'right_eye_x': data['right_gaze_point_on_display_area'][0],
                    'right_eye_y': data['right_gaze_point_on_display_area'][1],
                })

    def run(self):
        self.start_streaming()

class ExitWindow:
    def __init__(self, master):
        self.window = tk.Toplevel(master)
        self.window.title("制御パネル")
        self.window.geometry("300x100+100+100")
        self.window.attributes("-topmost", True)
        self.button = tk.Button(self.window, text="終了する", font=("Arial", 32), command=self.quit_app)
        self.button.pack(expand=True, fill='both')
        self.master = master

    def quit_app(self):
        logger.info("終了ボタンが押されました。アプリケーションを終了します。")
        self.master.quit()

# ===============================
#   メインアプリケーション (Tkinter版)
# ===============================
def main():
    global tobii_thread
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

    face_thread = FaceDetectionThread(speech_bubble)
    face_thread.daemon = True
    face_thread.start()

    tobii_thread = TobiiTrackingThread()
    tobii_thread.daemon = True
    tobii_thread.start()

    exit_window = ExitWindow(speech_bubble.root)

    try:
        speech_bubble.root.mainloop()
    finally:
        # mainloop終了後にtobiiの停止とCSV保存を行う
        tobii_thread.stop_streaming_and_save()
        logger.info("アプリケーションを終了しました。Tobiiデータを保存しました。")


if __name__ == "__main__":
    main()
