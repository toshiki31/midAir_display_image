import sys
import logging
import time
from PIL import Image, ImageTk
import tkinter as tk
import tkinter.font as tkfont
import screeninfo

# ===============================
#   visibility_expt.py
#   キーボード入力された文字を吹き出し字幕として提示するコード
# ===============================


# ===============================
#   定数・設定の定義
# ===============================
SPEECH_BUBBLE_IMG = "./images/speech-bubble1.png"  # 吹き出し画像

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

        # ポップアップウィンドウの生成
        self.popup = tk.Toplevel(self.root)
        self.popup.overrideredirect(True)
        self.popup.attributes("-topmost", True)
        self.popup.update_idletasks()
        self.popup.update()

        # モニター情報取得
        monitors = screeninfo.get_monitors()
       # スクリーン情報の取得（複数モニターがある場合は2番目のモニターを使用）
        if len(monitors) > 1:
            print("外部",monitors[1]) # 外部モニター情報のログ
            second_monitor = monitors[1]
            monitor_width = second_monitor.width
            monitor_height = second_monitor.height
            monitor_x = second_monitor.x
            monitor_y = second_monitor.y
        else:
            monitor_width = self.root.winfo_screenwidth()
            monitor_height = self.root.winfo_screenheight()
            monitor_x = 0
            monitor_y = 0

        # アスペクト比16:10でウィンドウサイズ
        aspect_ratio = 16 / 10
        popup_w = int(monitor_height * aspect_ratio)
        popup_h = monitor_height
        self.popup.geometry(f"{popup_w}x{popup_h}+{monitor_x}+{monitor_y}")
        
        # 保存用にウィンドウサイズ・スクリーン情報を保持
        self.popup_width = popup_w
        self.popup_height = popup_h
        self.monitor_height = monitor_height

        # 背景画像の読み込み
        bg_image_path = SPEECH_BUBBLE_IMG
        self.bg_image_orig = Image.open(bg_image_path)
        # 元画像サイズを取得
        orig_width, orig_height = self.bg_image_orig.size
        # ウィンドウ横幅に合わせた新しいサイズを計算（高さはアスペクト比に従う）
        new_width = popup_w
        new_height = int(orig_height * new_width / orig_width)
        self.bg_image = self.bg_image_orig.resize((new_width, new_height), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(self.bg_image)
        self.bg_height = new_height

        # Canvas を作成し、背景画像を配置
        self.canvas = tk.Canvas(self.popup, width=popup_w, height=popup_h, bg="black")
        self.canvas.pack()
        # 初期位置は y=0 として背景画像を配置
        self.bg_image_id = self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        # 字幕（テキスト）は背景画像の中央に配置（初期位置）
        self.text_id = self.canvas.create_text(popup_w / 2, new_height // 2,
                                                text="",
                                                font=("Arial", 200, "bold"),
                                                fill="black",
                                                width=popup_w - 100)
        self.text_font = tkfont.Font(family="Arial", size=200, weight="bold")

        self.last_time = time.time()
        self.poll_silence()

    def poll_silence(self):
        # 一定時間経過で非表示
        if time.time() - self.last_time >= 1:
            self.canvas.itemconfig(self.bg_image_id, state='hidden')
            self.canvas.itemconfig(self.text_id, state='hidden')
        else:
            self.canvas.itemconfig(self.bg_image_id, state='normal')
            self.canvas.itemconfig(self.text_id, state='normal')
        self.root.after(500, self.poll_silence)

    def update_text(self, text):
        self.last_time = time.time()
        self.canvas.itemconfig(self.bg_image_id, state='normal')
        self.canvas.itemconfig(self.text_id, state='normal')
        # テキスト幅調整
        max_w = self.popup_width - 100
        disp = text
        if self.text_font.measure(text) > max_w:
            for i in range(len(text)):
                sub = text[i:]
                if self.text_font.measure(sub) <= max_w:
                    disp = sub
                    break
        self.canvas.itemconfig(self.text_id, text=disp)
        self.root.update_idletasks()

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

    # キーボード入力用ウィジェット
    entry = tk.Text(bubble.root, width=30, height=5)
    entry.pack(pady=20)
    entry.focus_set()

    def update_label():
        text = entry.get("1.0", tk.END).strip()
        entry.delete("1.0", tk.END)
        bubble.update_text(text)
        entry.focus_set()

    # Shift+Enter で改行、それ以外の Enter で表示更新
    entry.bind("<Return>", lambda e: (update_label(), "break") if not (e.state & 0x0001) else None)

    bubble.root.mainloop()

if __name__ == '__main__':
    main()
