import asyncio
import pyaudio
import boto3
import logging
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

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

if __name__ == "__main__":
    try:
        asyncio.run(basic_transcribe())
    except KeyboardInterrupt:
        logger.info("Transcription stopped by user.")
