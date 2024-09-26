import cv2
import os
import numpy as np
import screeninfo
import csv
from datetime import datetime

def display_image(image_path, window_name, log_list):
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

    # ログに追加
    log_list.append((datetime.now().strftime("%Y-%m-%d %H:%M:%S"), image_path))

def write_log_to_csv(log_list):
    date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    filename=f'{date}_image_display_log.csv'
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['timestamp', 'image_path']
        writer = csv.writer(csvfile)
        writer.writerow(fieldnames)
        writer.writerows(log_list)

def main():
    monitors = screeninfo.get_monitors()
    num_monitors = len(monitors)

    if num_monitors < 2:
        monitor = monitors[0]
    else:
        monitor = monitors[1]
        
    screen_x = monitor.x
    screen_y = monitor.y

    print("Press 'a' to display comic-effect1.png")
    print("Press 's' to display comic-effect2.png")
    print("Press 'd' to display comic-effect3.png")
    print("Press 'f' to display comic-effect4.png")
    print("Press 'g' to display comic-effect5.png")
    print("Press 'h' to display comic-effect6.png")
    print("Press 'j' to display comic-effect7.png")
    print("Press 'k' to display comic-effect16.png")
    print("Press 'l' to display comic-effect8.png")
    print("Press 'z' to display comic-effect9.png")
    print("Press 'x' to display comic-effect10.png")
    print("Press 'c' to display comic-effect11.png")
    print("Press 'v' to display comic-effect12.png")
    print("Press 'b' to display comic-effect17.png")
    print("Press 'n' to display comic-effect14.png")
    print("Press 'm' to display comic-effect15.png")

    print("Press 'q' to quit")

    window_name = 'Image'
    # OpenCVのウィンドウを作成
    cv2.namedWindow(window_name)
    cv2.moveWindow(window_name, screen_x, screen_y)
    cv2.imshow(window_name, 255 * np.ones((100, 300), dtype=np.uint8))  # 初期の空白画像を表示

    log_list = []

    while True:
        key = cv2.waitKey(0) & 0xFF

        if key == ord('a'):
            display_image('./images/comic-effect1.png', window_name, log_list)
        elif key == ord('s'):
            display_image('images/comic-effect2.png', window_name, log_list)
        elif key == ord('d'):
            display_image('images/comic-effect3.png', window_name, log_list)
        elif key == ord('f'):
            display_image('images/comic-effect4.png', window_name, log_list)
        elif key == ord('g'):
            display_image('images/comic-effect5.png', window_name, log_list)
        elif key == ord('h'):
            display_image('images/comic-effect6.png', window_name, log_list)
        elif key == ord('j'):
            display_image('images/comic-effect7.png', window_name, log_list)
        elif key == ord('k'):
            display_image('images/comic-effect16.png', window_name, log_list)
        elif key == ord('l'):
            display_image('images/comic-effect8.png', window_name, log_list)
        elif key == ord('z'):
            display_image('images/comic-effect9.png', window_name, log_list)
        elif key == ord('x'):
            display_image('images/comic-effect10.png', window_name, log_list)
        elif key == ord('c'):
            display_image('images/comic-effect11.png', window_name, log_list)
        elif key == ord('v'):
            display_image('images/comic-effect12.png', window_name, log_list)
        elif key == ord('b'):
            display_image('images/comic-effect17.png', window_name, log_list)
        elif key == ord('n'):
            display_image('images/comic-effect14.png', window_name, log_list)
        elif key == ord('m'):
            display_image('images/comic-effect15.png', window_name, log_list)
        elif key == 13:  # Enter key
            display_image('images/black.png', window_name, log_list)
        elif key == ord('q'):
            print("Exiting...")
            break
        else:
            print(f"Key {chr(key)} pressed, no action assigned.")

    write_log_to_csv(log_list)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
