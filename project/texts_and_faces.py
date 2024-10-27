import asyncio
import pyaudio
import boto3
import logging
import cv2
import numpy as np
import os
import screeninfo
from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.handlers import TranscriptResultStreamHandler
from amazon_transcribe.model import TranscriptEvent
from botocore.exceptions import ClientError

# Transcription Configuration
SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2
CHANNEL_NUMS = 1
CHUNK_SIZE = 1024  # 1KB chunks
REGION = "ap-northeast-1"
TRANSCRIPT_LANGUAGE_CODE = "ja-JP"
MEDIA_ENCODING = "pcm"
COMPREHEND_LANGUAGE_CODE = "ja"

# detect faces Configuration 
scale_factor = .15
green = (0,255,0)
red = (0,0,255)
frame_thickness = 2
cap = cv2.VideoCapture(0)
session = boto3.Session(profile_name="rekognition")
rekognition = boto3.client('rekognition')

# Setup font
fontscale = 1.0
color = (0, 120, 238)
fontface = cv2.FONT_HERSHEY_DUPLEX


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

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def display_image(image_path, window_name):
    # 画像のパスをチェック
    abs_path = os.path.abspath(image_path)
    if not os.path.exists(abs_path):
        logger.error(f"Error: The path %abs_path% does not exist.")
        return
    
    # 画像を読み込み
    image = cv2.imread(abs_path)
    
    # 画像が正しく読み込まれたか確認
    if image is None:
        logger.error(f"Error: Unable to load image at %abs_path%")
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
                display_image('./images/black.png', window_name)
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
                self.analyze_sentiment(transcript_text)
    
    def analyze_sentiment(self, text):
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

    # PyAudio setup for live audio capture
    p = pyaudio.PyAudio()

    try:
        audio_stream = p.open(format=pyaudio.paInt16,
                              channels=CHANNEL_NUMS,
                              rate=SAMPLE_RATE,
                              input=True,
                              frames_per_buffer=CHUNK_SIZE)

        async def write_chunks():
            """Captures audio and sends it to Transcribe in chunks."""
            try:
                while True:
                    # 非同期で読み取り
                    audio_data = audio_stream.read(CHUNK_SIZE, exception_on_overflow=False)
                    await stream.input_stream.send_audio_event(audio_chunk=audio_data)
            except asyncio.CancelledError:
                pass
            finally:
                await stream.input_stream.end_stream()

        # Instantiate handler and start processing events
        handler = MyEventHandler(stream.output_stream)
        await asyncio.gather(write_chunks(), handler.handle_events())
    
    finally:
        # Ensure resources are cleaned up
        if audio_stream:
            audio_stream.stop_stream()
            audio_stream.close()
        p.terminate()

async def detect_faces():
    """Detect faces and analyze emotions using Rekognition."""
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        height, width, channels = frame.shape

        # Convert frame to jpg
        small = cv2.resize(frame, (int(width * scale_factor), int(height * scale_factor)))
        ret, buf = cv2.imencode('.jpg', small)

        # Detect faces in jpg
        faces = rekognition.detect_faces(Image={'Bytes':buf.tobytes()}, Attributes=['ALL'])

        # Draw rectangle around faces
        for face in faces['FaceDetails']:
            smile = face['Smile']['Value']
            cv2.rectangle(frame,
                        (int(face['BoundingBox']['Left']*width),
                        int(face['BoundingBox']['Top']*height)),
                        (int((face['BoundingBox']['Left']+face['BoundingBox']['Width'])*width),
                        int((face['BoundingBox']['Top']+face['BoundingBox']['Height'])*height)),
                        green if smile else red, frame_thickness)
            emotions = face['Emotions']
            i = 0
            # 感情によって表示画像を変える
            firstEmotion = emotions[0]
            logger.info("Detected emotion: %s", firstEmotion['Type'])
            if firstEmotion['Type'] == 'HAPPY':
                display_image('./images/comic-effect4.png', window_name)
            elif firstEmotion['Type'] == 'SURPRISED':
                display_image('./images/comic-effect1.png', window_name)
            elif firstEmotion['Type'] == 'CONFUSED':
                display_image('./images/comic-effect9.png', window_name)
            elif firstEmotion['Type'] == 'ANGRY':
                display_image('./images/comic-effect14.png', window_name)
            elif firstEmotion['Type'] == 'DISGUSTED':
                display_image('./images/comic-effect10.png', window_name)
            elif firstEmotion['Type'] == 'CALM':
                display_image('./images/comic-effect7.png', window_name)
            elif firstEmotion['Type'] == 'FEAR':
                display_image('./images/comic-effect18.png', window_name)
            elif firstEmotion['Type'] == 'SAD':
                display_image('./images/comic-effect17.png', window_name)
            else:
                display_image('/images/black.png', window_name)

            for emotion in emotions:
                cv2.putText(frame,
                            str(emotion['Type']) + ": " + str(emotion['Confidence']),
                            (25, 40 + (i * 25)),
                            fontface,
                            fontscale,
                            color)
                i += 1

        # Display the resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    threads1 = asyncio.create_task(basic_transcribe())
    threads2 = asyncio.create_task(detect_faces())

    threads2.start()
    threads1.start()
    
