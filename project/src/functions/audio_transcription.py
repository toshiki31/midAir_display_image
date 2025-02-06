import asyncio
import logging
import pyaudio
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.model import TranscriptEvent

# ===============================
#   定数・設定
# ===============================
TRANSCRIPT_LANGUAGE_CODE = "ja-JP"
SAMPLE_RATE = 16000
CHUNK_SIZE = 1024
MEDIA_ENCODING = "pcm"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===============================
#   音声認識スレッド (4メソッド版)
# ===============================
class AudioTranscriptionThread(QThread):
    """音声認識スレッド"""
    audio_detected = pyqtSignal()  # 音声が検出されたシグナル

    def __init__(self, comprehend_detect):
        """
        1) 初期化メソッド (__init__)
        """
        super().__init__()
        self.comprehend_detect = comprehend_detect
        self.pyaudio_instance = None
        self.audio_stream = None

    def setup_audio_input(self):
        """
        2) 音声入力を行うメソッド
           - マイク入力のストリームを開く
        """
        self.pyaudio_instance = pyaudio.PyAudio()
        self.audio_stream = self.pyaudio_instance.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE
        )
        logger.info("Audio stream opened.")

    def run(self):
        """
        QThread で走るメインエントリポイント。
        asyncio イベントループを起動し、transcribe_audio を実行
        """
        asyncio.run(self._main_async())

    async def _main_async(self):
        """
        run() から呼び出される非同期処理の実体
        """
        # まず音声入力をセットアップ
        self.setup_audio_input()
        # 次に Amazon Transcribe との接続と文字起こし処理を開始
        await self.transcribe_audio()

    async def transcribe_audio(self):
        """
        3) 音声データをAmazon Transcribeに送信して文字起こしスクリプトを受け取るメソッド
           - 非同期でマイク入力を読み取り、Transcribe に送信
           - Transcribe からの文字起こし結果を受信
           - 受信した文字起こし結果は analyze_transcript() で感情分析へ渡す
        """
        client = TranscribeStreamingClient(region="ap-northeast-1")

        # 音声認識用ストリームを開始
        stream = await client.start_stream_transcription(
            language_code=TRANSCRIPT_LANGUAGE_CODE,
            media_sample_rate_hz=SAMPLE_RATE,
            media_encoding=MEDIA_ENCODING,
        )
        logger.info("Started transcription stream.")

        async def write_chunks():
            """
            マイクから音声データを取得し続け、Transcribe に送信する内部関数
            """
            try:
                while True:
                    # ブロッキングI/Oを asyncio.to_thread でスレッドプール化
                    data = await asyncio.to_thread(
                        self.audio_stream.read, CHUNK_SIZE, exception_on_overflow=False
                    )
                    audio_np = np.frombuffer(data, dtype=np.int16)
                    # 音声レベルが閾値を超えたらシグナル発行
                    if np.abs(audio_np).mean() > 100:
                        self.audio_detected.emit()

                    # Transcribe へ音声を送信
                    await stream.input_stream.send_audio_event(audio_chunk=data)
            except asyncio.CancelledError:
                pass
            finally:
                # ストリーム終了を通知
                await stream.input_stream.end_stream()

        async def read_transcripts():
            """
            Transcribe からの文字起こし結果を読み取り、感情分析に渡す内部関数
            """
            async for transcript_event in stream.output_stream:
                if isinstance(transcript_event, TranscriptEvent):
                    results = transcript_event.transcript.results
                    for result in results:
                        for alt in result.alternatives:
                            transcript = alt.transcript
                            logger.info(f"Transcription: {transcript}")
                            # 受け取った文字起こしを感情分析に渡す
                            self.analyze_transcript(transcript)

        # 音声送信と文字起こし結果受信を同時に実行
        await asyncio.gather(write_chunks(), read_transcripts())

        # PyAudio の後処理
        self.audio_stream.stop_stream()
        self.audio_stream.close()
        self.pyaudio_instance.terminate()

    def analyze_transcript(self, transcript):
        """
        4) 受け取った文字起こしスクリプトを感情分析に渡すメソッド
        """
        self.comprehend_detect.detect_sentiment(transcript)
