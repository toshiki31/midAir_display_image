# .tobiienv内で実行してください

import tobii_research as tr
import time
import csv
from pynput import keyboard as kb

# アイトラッカー動作時間
waittime = 5

# アイトラッカー検出
found_eyetrackers = tr.find_all_eyetrackers()
my_eyetracker = found_eyetrackers[0]
gaze_data_list = []  # 視線データを格納するリスト


def gaze_data_callback(gaze_data):
    gaze_data_list.append(gaze_data)


# キーボードのsキーでストリーミングを開始

def toggle_streaming():
    print("視線トラッキングを開始します")
    my_eyetracker.subscribe_to(tr.EYETRACKER_GAZE_DATA, gaze_data_callback, as_dictionary=True)
    time.sleep(waittime)
    my_eyetracker.unsubscribe_from(tr.EYETRACKER_GAZE_DATA, gaze_data_callback)
    print("視線トラッキングを停止しました")

    # CSVファイルに視線データを書き込む
    with open('gaze_data.csv', 'w', newline='') as csvfile:
        fieldnames = ['left_eye_x', 'left_eye_y', 'right_eye_x', 'right_eye_y']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for data in gaze_data_list:
            left_eye_x = data['left_gaze_point_on_display_area'][0]
            left_eye_y = data['left_gaze_point_on_display_area'][1]
            right_eye_x = data['right_gaze_point_on_display_area'][0]
            right_eye_y = data['right_gaze_point_on_display_area'][1]

            # CSVに書き込み
            writer.writerow({
                'left_eye_x': left_eye_x,
                'left_eye_y': left_eye_y,
                'right_eye_x': right_eye_x,
                'right_eye_y': right_eye_y
            })


# pynputでキー入力を検知する関数
def on_press(key):
    try:
        if key.char == 's':
            toggle_streaming()
    except AttributeError:
        pass


print("'s'キーを押して視線トラッキングを開始します")

# キーボード入力を監視する
with kb.Listener(on_press=on_press) as listener:
    listener.join()