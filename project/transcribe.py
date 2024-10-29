import asyncio
import pyaudio
import boto3
import logging
import os
import cv2
import numpy as np
import screeninfo
import time
from botocore.exceptions import ClientError
from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.handlers import TranscriptResultStreamHandler
from amazon_transcribe.model import TranscriptEvent

# Configuration
SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2
CHANNEL_NUMS = 1
CHUNK_SIZE = 1024  # 1KB chunks
REGION = "ap-northeast-1"
TRANSCRIPT_LANGUAGE_CODE = "ja-JP"
MEDIA_ENCODING = "pcm"
COMPREHEND_LANGUAGE_CODE = "ja"
SILENT_SECONDS = 5

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 画像表示するモニターの設定
monitors = screeninfo.get_monitors()
num_monitors = len(monitors)
if num_monitors < 2:
    monitor = monitors[0]
else:
    monitor = monitors[1]
screen_x = monitor.x
screen_y = monitor.y
window_name = 'Image'
# OpenCVのウィンドウを作成
cv2.namedWindow(window_name)
cv2.moveWindow(window_name, screen_x, screen_y)
cv2.imshow(window_name, 255 * np.ones((100, 300), dtype=np.uint8))  # 初期の空白画像を表示
logger.info("Displaying initial image.")

def display_image(image_path, window_name):
    # 画像のパスをチェック
    abs_path = os.path.abspath(image_path)
    if not os.path.exists(abs_path):
        logger.error(f"Error: The path {abs_path} does not exist.")
        return
    
    # 画像を読み込み
    image = cv2.imread(abs_path)
    
    # 画像が正しく読み込まれたか確認
    if image is None:
        logger.error(f"Error: Unable to load image at {abs_path}")
        return
    
    # 16:10アスペクト比にリサイズ
    new_width = 3840
    new_height = int(new_width * 10 / 16)
    resized_image = cv2.resize(image, (new_width, new_height))

    # 画像をウィンドウに表示
    cv2.imshow(window_name, resized_image)

class ComprehendDetect:
    """Handles sentiment detection using Amazon Comprehend."""
    def __init__(self, comprehend_client):
        self.comprehend_client = comprehend_client

    def detect_sentiment(self, text, language_code):
        try:
            response = self.comprehend_client.detect_sentiment(
                Text=text, LanguageCode=language_code
            )
            sentiment = response["Sentiment"]
            logger.info("Detected sentiment: %s", sentiment)
            # 音声の感情分析結果によって画像を変更
            if sentiment == 'POSITIVE':
                display_image('./images/comic-effect4.png', window_name)
            elif sentiment == 'NEGATIVE':
                display_image('./images/comic-effect17.png', window_name)
            else:
                display_image('./images/comic-effect19.png', window_name)
            return sentiment
        except ClientError as error:
            logger.error("Error detecting sentiment: %s", error)
            raise

# Initialize Comprehend client
comp_detect = ComprehendDetect(boto3.client('comprehend', region_name=REGION))

class MyEventHandler(TranscriptResultStreamHandler):
    """Handles transcription events and performs sentiment analysis."""
    def __init__(self, output_stream):
        super().__init__(output_stream)
    
    async def handle_transcript_event(self, transcript_event: TranscriptEvent):
        """Processes transcription event and detects sentiment."""
        results = transcript_event.transcript.results
        for result in results:
            for alt in result.alternatives:
                transcript_text = alt.transcript
                logger.info("Transcription: %s", transcript_text)
                await self.analyze_sentiment(transcript_text)
    
    async def analyze_sentiment(self, text):
        """Performs sentiment analysis on transcribed text."""
        comp_detect.detect_sentiment(text, COMPREHEND_LANGUAGE_CODE)

async def basic_transcribe():
    """Captures audio from microphone and sends it to Amazon Transcribe."""
    client = TranscribeStreamingClient(region=REGION)

    # Start transcription stream
    stream = await client.start_stream_transcription(
        language_code=TRANSCRIPT_LANGUAGE_CODE,
        media_sample_rate_hz=SAMPLE_RATE,
        media_encoding=MEDIA_ENCODING,
    )
    logger.info("Started transcription stream.")

    # PyAudio setup for live audio capture
    p = pyaudio.PyAudio()

    try:
        audio_stream = p.open(format=pyaudio.paInt16,
                              channels=CHANNEL_NUMS,
                              rate=SAMPLE_RATE,
                              input=True,
                              frames_per_buffer=CHUNK_SIZE)
        logger.info("Audio stream opened.")

        async def write_chunks():
            """Captures audio and sends it to Transcribe in chunks."""
            global last_audio_time # 最後に音声を送信した時間をグローバル変数として扱う

            try:
                while True:
                    # 非同期で読み取り
                    audio_data = await asyncio.to_thread(audio_stream.read, CHUNK_SIZE, exception_on_overflow=False)
                    audio_np = np.frombuffer(audio_data, dtype=np.int16)
                    if np.abs(audio_np).mean() > 50:
                        last_audio_time = time.time() # 最後に音声を送信した時間を更新
                    await stream.input_stream.send_audio_event(audio_chunk=audio_data)
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.error(f"Error in write_chunks: {e}")
            finally:
                await stream.input_stream.end_stream()
                logger.info("Ended transcription stream.")

        # Instantiate handler and start processing events
        handler = MyEventHandler(stream.output_stream)
        await asyncio.gather(write_chunks(), handler.handle_events())

    except ClientError as e:
        logger.error(f"ClientError in basic_transcribe: {e}")
    except Exception as e:
        logger.error(f"Unexpected error in basic_transcribe: {e}")
    finally:
        # Ensure resources are cleaned up
        if 'audio_stream' in locals():
            audio_stream.stop_stream()
            audio_stream.close()
            logger.info("Audio stream closed.")
        p.terminate()
        logger.info("PyAudio terminated.")

async def monitor_audio():
    global last_audio_time
    while True:
        current_time = time.time()
        if current_time - last_audio_time > SILENT_SECONDS:
            logger.info("No audio detected for 5 seconds. Exiting.")
            display_image('./images/comic-effect8.png', window_name)
            last_audio_time = current_time # リセット
        await asyncio.sleep(1)

async def opencv_event_loop():
    """Handles OpenCV window events."""
    global last_audio_time

    while True:
        # Wait for 1 ms and process OpenCV events
        if cv2.waitKey(1) & 0xFF == ord('q'):
            logger.info("OpenCV window closed by user.")
            break
        await asyncio.sleep(0.01)  # Yield control to the event loop

async def main():
    # Display the initial image
    display_image('./images/black.png', window_name)

    # Run transcription and OpenCV event loop concurrently
    await asyncio.gather(
        basic_transcribe(),
        monitor_audio(),
        opencv_event_loop()
    )

if __name__ == "__main__":
    last_audio_time = time.time()
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Transcription stopped by user.")
    except Exception as e:
        logger.error(f"Unexpected error in main: {e}")
    finally:
        cv2.destroyAllWindows()
        logger.info("OpenCV windows closed.")