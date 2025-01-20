import cv2
import numpy as np
import boto3
import os
import screeninfo
from PIL import Image, ImageSequence
import threading

def display_image(image_path, window_name):
    # 画像のパスをチェック
    abs_path = os.path.abspath(image_path)
    if not os.path.exists(abs_path):
        print(f"Error: The path {abs_path} does not exist.")
        return
    
    # 画像を読み込み
    image = cv2.imread(abs_path)
    
    # 画像が正しく読み込まれたか確認
    if image is None:
        print(f"Error: Unable to load image at {abs_path}")
        return
    resized_image = cv2.resize(image, (WINDOW_WIDTH, WINDOW_HEIGHT))

    # 画像をウィンドウに表示
    cv2.imshow(window_name, resized_image)

    # # ログに追加
    # log_list.append((datetime.now().strftime("%Y-%m-%d %H:%M:%S"), image_path))

def read_gif(gifList):
    gif = Image.open(gifList[0])
    frames = []
    for frame in ImageSequence.Iterator(gif):
        # RGB -> BGRに変換
        frame_array = np.array(frame.convert('RGB'))[..., ::-1]
        frames.append(frame_array)
    return np.array(frames)


def show_gif(gif, window_name):
    for t in range(len(gif)):
        resized_frame = cv2.resize(gif[t], (WINDOW_WIDTH, WINDOW_HEIGHT))
        cv2.imshow(window_name, resized_frame)
        cv2.waitKey(10)

def play_video_in_main(movie, window_name):
    """
    動画をメインスレッドで再生する関数。
    """
    cap = cv2.VideoCapture(movie)
    if not cap.isOpened():
        print(f"Error: Cannot open video {movie}")
        return

    while True:
        ret, img = cap.read()
        if not ret:
            break  # 動画再生終了
        
        img = cv2.resize(img, (WINDOW_WIDTH, WINDOW_HEIGHT))
        cv2.imshow(window_name, img)

        # OpenCV のイベント処理を確保
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    print(f"Finished playing video: {movie}")

# Setup
scale_factor = .15
green = (0,255,0)
red = (0,0,255)
frame_thickness = 2
cap = cv2.VideoCapture(0)
session = boto3.Session(profile_name="rekognition")
rekognition = boto3.client('rekognition')
WINDOW_WIDTH = 3840
WINDOW_HEIGHT = int(WINDOW_WIDTH * 10 / 16)

HAPPY_ANIMATION = ["./gifs/happy_animation.gif", 4]
SURPRISED_ANIMATION = ["./gifs/surprised_animation.gif", 24]
happy_gif = read_gif(HAPPY_ANIMATION)
surprised_gif = read_gif(SURPRISED_ANIMATION)

HAPPY_MOVIE = "./movies/happy_movie.mp4"
SURPRISED_MOVIE = "./movies/surprised_movie.mp4"


# font-size
fontscale = 1.0
# font-color (B, G, R)
color = (0, 120, 238)
# font
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

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame from camera.")
        break

    height, width, channels = frame.shape

    # Convert frame to jpg
    small = cv2.resize(frame, (int(width * scale_factor), int(height * scale_factor)))
    ret, buf = cv2.imencode('.jpg', small)

    # Detect faces in jpg
    faces = rekognition.detect_faces(Image={'Bytes': buf.tobytes()}, Attributes=['ALL'])

    # Draw rectangle around faces
    for face in faces['FaceDetails']:
        smile = face['Smile']['Value']
        cv2.rectangle(
            frame,
            (int(face['BoundingBox']['Left'] * width), int(face['BoundingBox']['Top'] * height)),
            (int((face['BoundingBox']['Left'] + face['BoundingBox']['Width']) * width),
             int((face['BoundingBox']['Top'] + face['BoundingBox']['Height']) * height)),
            green if smile else red, frame_thickness
        )

        emotions = face['Emotions']
        firstEmotion = emotions[0]
        if firstEmotion['Type'] == 'HAPPY':
            play_video_in_main(HAPPY_MOVIE, window_name)
        elif firstEmotion['Type'] == 'SURPRISED':
            play_video_in_main(SURPRISED_MOVIE, window_name)
        else:
            display_image('./images/black.png', window_name)

        # Draw emotions on the frame
        for i, emotion in enumerate(emotions):
            cv2.putText(
                frame,
                f"{emotion['Type']}: {emotion['Confidence']:.2f}",
                (25, 40 + (i * 25)),
                fontface,
                fontscale,
                color
            )

    # Display the camera frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()