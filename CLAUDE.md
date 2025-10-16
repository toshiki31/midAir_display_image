# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python-based interactive communication system that displays manga-style visual effects (comic effects) based on:
- Real-time facial expression recognition (using AWS Rekognition)
- Speech-to-text transcription (using AWS Transcribe)
- Sentiment analysis (using AWS Comprehend)
- Eye tracking integration (using Tobii eye tracker)

The system displays appropriate comic effects (images) on a secondary monitor based on detected emotions and speech sentiment, creating an augmented communication experience.

## Environment Setup

**Python Version Management:**
- Uses `pyenv` for Python version management
- Current version is stored in `.python-version` file
- Check version: `python -V`
- Install specific version: `pyenv install [version]`
- Set local version: `pyenv local [version]`

**Virtual Environment:**
- Located at `project/.venv/`
- Activate: `source project/.venv/bin/activate`
- Deactivate: `deactivate`
- Install dependencies: `pip install -r project/requirements.txt`
- Update requirements: `pip freeze > project/requirements.txt`

**Running Programs:**
```bash
python project/src/[filename].py
```

## Key Dependencies

- **AWS Services:** boto3, amazon-transcribe (requires AWS credentials configured)
- **Computer Vision:** opencv-python, opencv-python-headless
- **Audio:** PyAudio
- **GUI:** PyQt5, tkinter (for different UI implementations)
- **Eye Tracking:** tobii_research (for Tobii integration)
- **Data Analysis:** pandas, matplotlib, numpy

## Architecture Overview

### Core Components

**1. Facial Expression Recognition (`detect_faces*.py`)**
- Captures webcam video using OpenCV
- Sends frames to AWS Rekognition for emotion detection
- Maps emotions (HAPPY, SURPRISED, CONFUSED, ANGRY, DISGUSTED, CALM, FEAR, SAD) to specific comic effect images
- Displays corresponding images on a secondary monitor

**2. Speech Recognition + Sentiment (`transcribe.py`, `texts_and_faces*.py`)**
- Uses AWS Transcribe Streaming for real-time Japanese speech-to-text
- Analyzes transcribed text with AWS Comprehend for sentiment
- Combines facial expressions and text sentiment for multimodal emotion detection
- Implements silence detection (3s, 5s, 7s thresholds) to trigger "thinking" animations

**3. Translation + Speech Bubbles (`translated_speech_bubble*.py`, `display_translated_scripts*.py`)**
- Real-time speech recognition and translation between languages
- Displays translated text in manga-style speech bubbles
- `translated_speech_bubble2*.py`: Dynamic speech bubble positioning based on face detection
- Supports projector display for conversation experiments

**4. Eye Tracking Integration (`*_tobii.py` files)**
- Files ending in `_tobii.py` integrate Tobii eye tracker
- Logs gaze data to CSV files with timestamps
- Used for conversation experiments and visibility studies
- Located primarily in `project/src/` with analysis tools in `project/src/analysis/`

**5. Experiment Modules**
- `visibility_expt/`: Studies on comic effect visibility
- `analysis/gaze_heatmap.py`: Generates heatmaps from eye tracking data

### Display System

- Uses `screeninfo` to detect multiple monitors
- Typically displays comic effects on the second monitor (or first if only one exists)
- Comic effect images stored in `project/images/` directory
- Images include emotion-based effects (comic-effect1.png through comic-effect18.png) and speech bubbles

### File Variants

Scripts come in multiple variants:
- **Base version** (e.g., `detect_faces.py`): PNG/static image processing
- **`_gif` version**: Animated GIF support
- **`_movie` version**: Video processing with OpenCV
- **`_movie_pyqt` version**: Video processing with PyQt5 GUI
- **`_tobii` version**: Includes Tobii eye tracking integration
- **`_test` version**: Testing/experimental implementations

## AWS Configuration

**Required AWS Services:**
- Rekognition (facial analysis)
- Transcribe (speech-to-text)
- Comprehend (sentiment analysis)
- Translate (language translation)

**AWS Profile:**
- Uses named profile "rekognition" in some scripts
- Configure with: `aws configure --profile rekognition`
- Region: `ap-northeast-1` (Tokyo)

## Common Workflows

**Running facial expression detection:**
```bash
source project/.venv/bin/activate
python project/src/detect_faces_movie_pyqt.py
```

**Running speech bubble translation (for experiments):**
```bash
source project/.venv/bin/activate
python project/src/translated_speech_bubble2_tobii.py
```

**Multimodal emotion detection (text + face):**
```bash
source project/.venv/bin/activate
python project/src/texts_and_faces_movie.py
```

**Muting microphone for expression-only mode:**
System Settings > Sound > Set input volume to minimum

## Code Architecture Patterns

**Async Pattern:**
Scripts using AWS Transcribe implement async/await patterns with `asyncio` for streaming audio processing.

**Multi-threading:**
Tobii integration scripts use threading to separate:
- Main GUI/display thread (tkinter/PyQt)
- Audio transcription thread
- Video capture thread (when applicable)

**Image Display:**
Common pattern across all scripts:
```python
def display_image(image_path, window_name):
    # Load image
    # Resize to 16:10 aspect ratio (3840x2400)
    # Display on secondary monitor using screeninfo
```

**State Management:**
Recent scripts use `SystemState` classes to manage shared state (e.g., `last_audio_time`) across async tasks.

**Window Managers:**
`ImageWindowManager` classes encapsulate monitor detection and image display logic.
