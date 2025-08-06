import sys
import logging
import time
import random
import string
from PIL import Image, ImageTk
import tkinter as tk
import tkinter.font as tkfont
import screeninfo

# ===============================
#   visibility_expt.py
#   任意のキー押下でランダム8文字を吹き出しに表示
# ===============================

# ===============================
#   定数・設定の定義
# ===============================
SPEECH_BUBBLE_IMG = "./images/speech-bubble1.png"  # 吹き出し画像
FONT_SIZE = 200

# ロガーセットアップ
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===============================
#   吹き出し表示クラス
# ===============================
class SpeechBubble:
    def __init__(self):
        # メインウィンドウ
        self.root = tk.Tk()

        # ポップアップウィンドウを生成（枠なし＋常に手前）
        self.popup = tk.Toplevel(self.root)
        self.popup.overrideredirect(True)
        self.popup.attributes("-topmost", True)
        self.popup.update_idletasks()
        self.popup.update()

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
        aspect_ratio = 16 / 10
        popup_w = int(monitor_height * aspect_ratio)
        popup_h = monitor_height
        self.popup.geometry(f"{popup_w}x{popup_h}+{monitor_x}+{monitor_y}")

        # Canvas／吹き出し背景のサイズ情報を保持
        self.popup_width  = popup_w
        self.popup_height = popup_h
        self.monitor_height = monitor_height

        # 吹き出し画像の読み込み＆リサイズ
        bg_image_path = SPEECH_BUBBLE_IMG
        self.bg_image_orig = Image.open(bg_image_path)
        orig_width, orig_height = self.bg_image_orig.size
        new_width  = popup_w
        new_height = int(orig_height * new_width / orig_width)
        self.bg_image = self.bg_image_orig.resize(
            (new_width, new_height),
            Image.Resampling.LANCZOS
        )
        self.photo     = ImageTk.PhotoImage(self.bg_image)
        self.bg_height = new_height

        # Canvas に背景画像を配置
        self.canvas = tk.Canvas(self.popup, width=popup_w, height=popup_h, bg="black")
        self.canvas.pack()
        self.bg_id = self.canvas.create_image(
            0, 0,
            image=self.photo,
            anchor=tk.NW
        )

        # フォント準備＆テキスト配置
        self.font_size = FONT_SIZE
        self.text_font = tkfont.Font(family="Arial", size=self.font_size, weight="bold")
        self.text_id = self.canvas.create_text(
            popup_w/2, new_height//2,
            text="",
            font=self.text_font,
            fill="black",
            width=popup_w-100
        )

        # 自動隠蔽タイマー開始
        self.last_time = time.time()
        self._poll()

    def _poll(self):
        # 3秒無操作で非表示、そうでなければ表示
        if time.time() - self.last_time >= 3:
            self.canvas.itemconfig(self.bg_id, state='hidden')
            self.canvas.itemconfig(self.text_id, state='hidden')
        else:
            self.canvas.itemconfig(self.bg_id, state='normal')
            self.canvas.itemconfig(self.text_id, state='normal')
        self.root.after(500, self._poll)

    def update_text(self, text: str):
        # 表示更新＆自動隠蔽タイマーリセット
        self.last_time = time.time()
        self.canvas.itemconfig(self.bg_id, state='normal')
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
        self.font_size = new
        self.text_font.configure(size=self.font_size)
        self.canvas.itemconfig(self.text_id, font=self.text_font)
        logger.info(f"フォントサイズを {self.font_size} に変更")

    def show(self):
        self.popup.deiconify()
        self.root.update()

    def hide(self):
        self.popup.withdraw()
        self.root.update()


# ===============================
#   メイン
# ===============================
def main():
    bubble = SpeechBubble()
    bubble.show()

    # Enterキー押下でランダム8文字を表示
    def create_random_text(event):
        rand_str = ''.join(
            random.choices(string.ascii_letters + string.digits, k=8)
        )
        bubble.update_text(rand_str)
        logger.info(f"表示文字: {rand_str}")

    # Enter キー押下でランダム文字列生成
    bubble.root.bind_all("<Return>", create_random_text)

    # a/sでフォントサイズを増減
    bubble.root.bind_all("a",   lambda e: bubble.change_font_size(+50))
    bubble.root.bind_all("s", lambda e: bubble.change_font_size(-50))

    bubble.root.mainloop()

if __name__ == '__main__':
    main()
