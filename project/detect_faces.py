import cv2
import numpy as np
import boto3
import os
import screeninfo

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
    
    # 16:10アスペクト比にリサイズ
    height, width = image.shape[:2]
    new_width = 3840
    new_height = int(new_width * 10 / 16)
    resized_image = cv2.resize(image, (new_width, new_height))

    # 画像をウィンドウに表示
    cv2.imshow(window_name, resized_image)

    # # ログに追加
    # log_list.append((datetime.now().strftime("%Y-%m-%d %H:%M:%S"), image_path))

# Setup
scale_factor = .15
green = (0,255,0)
red = (0,0,255)
frame_thickness = 2
cap = cv2.VideoCapture(0)
session = boto3.Session(profile_name="rekognition")
rekognition = boto3.client('rekognition')

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

while(True):

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
            display_image('./images/comic-effect13.png', window_name)
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

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()