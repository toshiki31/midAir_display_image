import sys
import logging
import time
import random
import string
import threading
from datetime import datetime
from PIL import Image, ImageTk
import tkinter as tk
import tkinter.font as tkfont
import screeninfo
import tobii_research as tr
import csv

# ===============================
#   定数・設定の定義
# ===============================
SPEECH_BUBBLE_IMG = "./images/speech-bubble1.png"  # 吹き出し画像
FONT_SIZE = 100
LIMIT_TIME = 60
DELTA=25

# ロガーセットアップ
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===============================
#   吹き出し表示クラス
# ===============================
class SpeechBubble:
    def __init__(self):
        # メインウィンドウを枠なし全画面に
        self.root = tk.Tk()
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        self.root.update_idletasks()
        self.root.update()

        # モニター情報取得（外部ディスプレイがあれば monitors[1] を使用）
        monitors = screeninfo.get_monitors()
        if len(monitors) > 1:
            second_monitor = monitors[1]
            monitor_width  = second_monitor.width
            monitor_height = second_monitor.height
            monitor_x      = second_monitor.x
            monitor_y      = second_monitor.y
        else:
            monitor_width  = self.root.winfo_screenwidth()
            monitor_height = self.root.winfo_screenheight()
            monitor_x = monitor_y = 0

        # アスペクト比16:10でウィンドウサイズを計算
        aspect_ratio = 16 / 9
        popup_w = int(monitor_height * aspect_ratio)
        popup_h = monitor_height
        self.root.geometry(f"{popup_w}x{popup_h}+{monitor_x}+{monitor_y}")

        self.root.lift()         # 最前面に
        self.root.focus_force()  # キーボードフォーカスを強制
        self.root.grab_set()     # 入力をこのウィンドウに集中

        # ウィンドウサイズ情報を保持
        self.popup_width  = popup_w
        self.popup_height = popup_h
        self.monitor_height = monitor_height

        # 吹き出し画像の読み込み＆リサイズ
        self.bg_image_orig = Image.open(SPEECH_BUBBLE_IMG)
        orig_width, orig_height = self.bg_image_orig.size
        new_width  = popup_w
        new_height = int(orig_height * new_width / orig_width)
        self.bg_image = self.bg_image_orig.resize(
            (new_width, new_height),
            Image.Resampling.LANCZOS
        )
        self.photo     = ImageTk.PhotoImage(self.bg_image)
        self.bg_height = new_height

        self.bg_y = 0  # 背景画像の Y 座標を保持する変数

        # Canvas に背景画像を配置
        self.canvas = tk.Canvas(self.root, width=popup_w, height=popup_h, bg="black")
        self.canvas.pack()
        self.bg_id = self.canvas.create_image(
            0, self.bg_y, image=self.photo, anchor=tk.NW
        )

        # フォント準備＆テキスト配置
        self.font_size = FONT_SIZE
        self.text_font = tkfont.Font(family="Arial", size=self.font_size, weight="bold")
        self.text_id = self.canvas.create_text(
            popup_w/2, self.bg_y + new_height//2,
            text="",
            font=self.text_font,
            fill="black",
            width=popup_w-100
        )

        # 自動隠蔽タイマー開始
        self.last_time = time.time()
        self._poll()

    def _poll(self):
        # 変数で与えられた秒数秒無操作で非表示、そうでなければ表示
        if time.time() - self.last_time >= LIMIT_TIME:
            self.canvas.itemconfig(self.bg_id, state='hidden')
            self.canvas.itemconfig(self.text_id, state='hidden')
        else:
            self.canvas.itemconfig(self.bg_id, state='normal')
            self.canvas.itemconfig(self.text_id, state='normal')
        self.root.after(500, self._poll)

    def update_text(self, text: str):
        # 表示更新＆自動隠蔽タイマーリセット
        self.last_time = time.time()
        self.canvas.itemconfig(self.bg_id,   state='normal')
        self.canvas.itemconfig(self.text_id, state='normal')

        # 幅がはみ出る場合は右側を切り詰め
        max_w = self.canvas.winfo_width() - 100
        if self.text_font.measure(text) > max_w:
            for i in range(len(text)):
                sub = text[i:]
                if self.text_font.measure(sub) <= max_w:
                    text = sub
                    break

        self.canvas.itemconfig(self.text_id, text=text)
        self.root.update_idletasks()

    def change_font_size(self, delta):
        """フォントサイズを増減し、Canvas上のテキストを更新"""
        new = max(50, min(300, self.font_size + delta))
        if new == self.font_size:
            return
        old = self.font_size
        self.font_size = new
        self.text_font.configure(size=self.font_size)
        self.canvas.itemconfig(self.text_id, font=self.text_font)
        logger.info(f"フォントサイズ: {old} → {self.font_size}")

    def move_bubble(self, delta_y):
        """
        背景画像とテキストを delta_y 分上下に移動させる。
        画面外にはみ出さないよう clamp しています。
        """
        old_y = self.bg_y
        new_y = max(0, min(self.popup_height - self.bg_height, self.bg_y + delta_y))
        self.bg_y = new_y
        self.canvas.coords(self.bg_id, 0, self.bg_y)
        self.canvas.coords(
            self.text_id,
            self.popup_width/2,
            self.bg_y + self.bg_height//2
        )
        self.root.update_idletasks()
        logger.info(f"バブル位置 Y: {old_y} → {new_y}")

    def show(self):
        self.root.deiconify()
        self.root.update()

    def hide(self):
        self.root.withdraw()
        self.root.update()


# ===============================
#   Tobii 視線計測スレッド
# ===============================
class TobiiTrackingThread(threading.Thread):
    def __init__(self, bubble):   # bubbleを受け取る
        super().__init__()
        self.gaze_data_list = []
        self.running = False
        self.start_time = None
        self.bubble = bubble  # ← bubbleを保持

        found = tr.find_all_eyetrackers()
        if not found:
            raise RuntimeError("アイトラッカーが見つかりません")
        self.eyetracker = found[0]

    def gaze_callback(self, gaze_data):
        self.gaze_data_list.append(gaze_data)

    def start_streaming(self):
        if self.running:
            logger.info("Tobii: すでに計測中です")
            return
        self.running = True
        self.start_time = time.time()
        self.gaze_data_list = []  # ← 新しい計測ではリストをリセット
        logger.info("Tobii: 計測開始")
        self.eyetracker.subscribe_to(
            tr.EYETRACKER_GAZE_DATA,
            self.gaze_callback,
            as_dictionary=True
        )

    def stop_and_save(self):
        if not self.running:
            logger.info("Tobii: 計測中ではありません")
            return
        self.running = False
        self.eyetracker.unsubscribe_from(
            tr.EYETRACKER_GAZE_DATA,
            self.gaze_callback
        )

        elapsed = time.time() - self.start_time if self.start_time else 0
        logger.info(f"Tobii: 計測時間 {elapsed:.2f} 秒")

        # Bubbleのフォントサイズと位置を含めたファイル名
        filename = datetime.now().strftime(
            f"visibility_expt_%Y%m%d_%H%M%S_font{self.bubble.font_size}_y{self.bubble.bg_y}.csv"
        )

        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['left_x','left_y','right_x','right_y'])
            for d in self.gaze_data_list:
                lx, ly = d['left_gaze_point_on_display_area']
                rx, ry = d['right_gaze_point_on_display_area']
                writer.writerow([lx, ly, rx, ry])
        logger.info(f"Tobii: データを {filename} に保存")

    def toggle_streaming(self):
        if self.running:
            self.stop_and_save()
        else:
            self.start_streaming()


# ===============================
#   メイン
# ===============================
def main():
    bubble = SpeechBubble()
    bubble.show()

    tobii_thread = TobiiTrackingThread(bubble)
    tobii_thread.daemon = True

    # キー操作バインド
    bubble.root.bind_all("<Return>", lambda e: bubble.update_text(
        ''.join(random.choices(string.ascii_letters + string.digits, k=8))
    ))
    bubble.root.bind_all("a", lambda e: bubble.change_font_size(+DELTA))
    bubble.root.bind_all("s", lambda e: bubble.change_font_size(-DELTA))
    bubble.root.bind_all("<Up>",   lambda e: bubble.move_bubble(-50))
    bubble.root.bind_all("<Down>", lambda e: bubble.move_bubble(+50))

    # Tobii用キー: fキーで開始・停止をトグル
    bubble.root.bind_all("f", lambda e: (tobii_thread.toggle_streaming(), "break"))

    bubble.root.mainloop()

if __name__ == "__main__":
    main()
