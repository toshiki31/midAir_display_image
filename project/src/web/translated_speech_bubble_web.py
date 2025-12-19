import sys
import asyncio
import logging
import time
import cv2
import boto3
import numpy as np
import pyaudio
import socket
from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.handlers import TranscriptResultStreamHandler
from amazon_transcribe.model import TranscriptEvent
import threading
import argparse
from flask import Flask, render_template
from flask_socketio import SocketIO, emit

# ===============================
#   定数・設定の定義
# ===============================
SAMPLE_RATE = 16000
CHUNK_SIZE = 1024
MEDIA_ENCODING = "pcm"
COMPREHEND_LANGUAGE_CODE = "ja"
TRANSCRIPT_LANGUAGE_CODE = "ja-JP"  # 初期値
TARGET_LANGUAGE_CODE = "en-US"    # 初期値
REGION = "ap-northeast-1"
SILENCE_THRESHOLD = 1.0  # 1秒無音で非表示

# ロガーセットアップ
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask アプリケーション
app = Flask(__name__)
app.config['SECRET_KEY'] = 'comic-effect-secret'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# グローバル状態管理
class SystemState:
    def __init__(self):
        self.last_transcript_time = time.time()
        self.current_text = ""
        self.current_y_position = 0
        self.is_visible = False
        self.connected_clients = 0

system_state = SystemState()

# グローバルスレッド管理
audio_thread = None
face_thread = None
silence_thread = None


# ===============================
#   コマンドライン引数の解析
# ===============================
def parse_arguments():
    """コマンドライン引数を解析"""
    parser = argparse.ArgumentParser(
        description='Web-based Speech Bubble Translation System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Start server (language selection in web UI)
  python %(prog)s

  # Specify custom port
  python %(prog)s --port 8080

  # Specify custom host and port
  python %(prog)s --host 192.168.1.100 --port 8080
        '''
    )

    parser.add_argument(
        '--host',
        default='0.0.0.0',
        help='Server host (default: 0.0.0.0)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=5010,
        help='Server port (default: 5010)'
    )

    return parser.parse_args()


# ===============================
#   Flask ルート
# ===============================
@app.route('/')
def index():
    return render_template('speech_bubble.html')


# ===============================
#   SocketIO イベントハンドラ
# ===============================
@socketio.on('connect')
def handle_connect():
    system_state.connected_clients += 1
    logger.info(f'Client connected. Total clients: {system_state.connected_clients}')
    # 現在の状態を新しいクライアントに送信
    if system_state.is_visible:
        emit('update_text', {'text': system_state.current_text})
        emit('update_position', {'y': system_state.current_y_position})
        emit('show_bubble')
    else:
        emit('hide_bubble')


@socketio.on('disconnect')
def handle_disconnect():
    system_state.connected_clients -= 1
    logger.info(f'Client disconnected. Total clients: {system_state.connected_clients}')


@socketio.on('client_ready')
def handle_client_ready(data):
    logger.info(f'Client ready with screen: {data.get("screen_width")}x{data.get("screen_height")}')


@socketio.on('select_language')
def handle_language_selection(data):
    """言語選択イベントを処理し、システムスレッドを起動"""
    global TRANSCRIPT_LANGUAGE_CODE, TARGET_LANGUAGE_CODE
    global audio_thread, face_thread, silence_thread

    # 言語コードを更新
    TRANSCRIPT_LANGUAGE_CODE = data.get('source_lang', 'ja-JP')
    TARGET_LANGUAGE_CODE = data.get('target_lang', 'en-US')
    logger.info(f'Language selected: {TRANSCRIPT_LANGUAGE_CODE} -> {TARGET_LANGUAGE_CODE}')

    # スレッドが既に起動しているかチェック
    if audio_thread is None or not audio_thread.is_alive():
        logger.info("Starting audio transcription thread...")
        audio_thread = AudioTranscriptionThread()
        audio_thread.start()

    if face_thread is None or not face_thread.is_alive():
        logger.info("Starting face detection thread...")
        face_thread = FaceDetectionThread()
        face_thread.start()

    if silence_thread is None or not silence_thread.is_alive():
        logger.info("Starting silence detection thread...")
        silence_thread = SilenceDetectionThread()
        silence_thread.start()

    # クライアントにシステム起動完了を通知
    emit('system_started', {
        'status': 'success',
        'source_lang': TRANSCRIPT_LANGUAGE_CODE,
        'target_lang': TARGET_LANGUAGE_CODE
    })
    logger.info("System started successfully")


# ===============================
#   音声認識スレッド
# ===============================
class AudioTranscriptionThread(threading.Thread):
    """音声認識スレッド"""
    def __init__(self):
        super().__init__()
        self.loop = None
        self.daemon = True

    def run(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self.transcribe_stream())

    async def transcribe_stream(self):
        audio_stream = None
        p = None
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

            handler = MyEventHandler(stream.output_stream)
            await asyncio.gather(write_chunks(), handler.handle_events())

        except Exception as e:
            logger.error(f"Error in transcribe_stream: {e}")
        finally:
            if audio_stream is not None:
                audio_stream.stop_stream()
                audio_stream.close()
            if p is not None:
                p.terminate()


# ===============================
#   音声認識 -> 翻訳 -> WebSocket配信
# ===============================
class MyEventHandler(TranscriptResultStreamHandler):
    def __init__(self, output_stream):
        super().__init__(output_stream)

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

                # 状態を更新
                system_state.last_transcript_time = time.time()
                system_state.current_text = translated
                system_state.is_visible = True

                # WebSocketで全クライアントに送信
                socketio.emit('update_text', {'text': translated})
                socketio.emit('show_bubble')


# ===============================
#   顔検出スレッド
# ===============================
class FaceDetectionThread(threading.Thread):
    """
    カメラ映像から顔検出を行い、検出された顔の上端に基づいて
    吹き出しの位置を更新するスレッドです。
    """
    def __init__(self, offset=600):
        super().__init__()
        self.offset = offset
        self.running = True
        self.daemon = True
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def run(self):
        cap = cv2.VideoCapture(0)
        # iPhoneの典型的な解像度に合わせる（例: iPhone 14 Pro = 1179x2556）
        # ただし、カメラは通常の解像度で取得
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        while self.running:
            ret, frame = cap.read()
            if not ret:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            if len(faces) > 0:
                (x, y, w, h) = faces[0]
                face_top = int(y)  # numpy.int64をintに変換
                # 位置調整
                new_y = face_top - self.offset
                if new_y < 0:
                    new_y = 0

                # 状態を更新
                system_state.current_y_position = new_y

                # WebSocketで位置情報を送信
                socketio.emit('update_position', {'y': int(new_y)})  # intに変換してJSON serializableにする
                logger.info(f"Face detected at y={face_top}, bubble position={new_y}")

            time.sleep(0.1)

        cap.release()


# ===============================
#   無音検出スレッド
# ===============================
class SilenceDetectionThread(threading.Thread):
    """無音時に吹き出しを非表示にするスレッド"""
    def __init__(self):
        super().__init__()
        self.running = True
        self.daemon = True

    def run(self):
        while self.running:
            time.sleep(0.5)

            # 最後のトランスクリプトから一定時間経過したら非表示
            if system_state.is_visible:
                elapsed = time.time() - system_state.last_transcript_time
                if elapsed >= SILENCE_THRESHOLD:
                    system_state.is_visible = False
                    socketio.emit('hide_bubble')
                    logger.info("Hiding bubble due to silence")


# ===============================
#   ローカルIPアドレス取得
# ===============================
def get_local_ip():
    """ローカルIPアドレスを取得"""
    try:
        # ダミー接続でローカルIPを取得
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception:
        return "127.0.0.1"


# ===============================
#   メインアプリケーション
# ===============================
def main():
    # コマンドライン引数を解析
    args = parse_arguments()

    # ローカルIPアドレスを表示
    local_ip = get_local_ip()
    print("\n" + "="*60)
    print("🎉 Server is running!")
    print("="*60)
    print(f"\n📱 iPhoneのブラウザで以下のURLにアクセスしてください:")
    print(f"\n   http://{local_ip}:{args.port}")
    print(f"\n   (または http://localhost:{args.port} でローカル確認)")
    print(f"\n💡 ブラウザで言語を選択してシステムを開始してください")
    print("\n" + "="*60 + "\n")

    # Flaskサーバーを起動（メインスレッドで実行）
    socketio.run(app, host=args.host, port=args.port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
