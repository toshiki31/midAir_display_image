# ===============================
#   視線が画面下方に来たら、外部モニター上の吹き出しにスローン文字（視力検査で使う10種類の文字）4文字をランダム表示し、
#   位置も5パターンでランダムに動かす。計測の開始/停止で視線データをCSVに保存できるコード
# ===============================

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
FONT_SIZE = 150
LIMIT_TIME = 60
DELTA = 25
BOTTOM_OFFSET = 400  # 「左下/右下」は基準Yからこの分だけ下げる
X_OFFSET = 200 # モニターの位置を下げた時に拡大されるからOffsetつける
Y_OFFSET = 300

# スローン文字（Sloan letters）
SLOAN_LETTERS = "CDHKNORSVZ"

def random_sloan(n: int = 4) -> str:
    """スローン文字から n 文字のランダム文字列を生成"""
    return ''.join(random.choices(SLOAN_LETTERS, k=n))

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

        # アスペクト比16:9でウィンドウサイズを計算
        aspect_ratio = 16 / 9
        popup_w = int(monitor_height * aspect_ratio)
        popup_h = monitor_height
        self.root.geometry(f"{popup_w}x{popup_h}+{monitor_x}+{monitor_y}")

        self.root.lift()
        self.root.focus_force()
        self.root.grab_set()

        # ウィンドウサイズ情報を保持
        self.popup_width  = popup_w
        self.popup_height = popup_h

        # 吹き出し画像の読み込み（高さは従来のまま、幅だけ1/3に）
        self.bg_image_orig = Image.open(SPEECH_BUBBLE_IMG)
        orig_width, orig_height = self.bg_image_orig.size
        previous_height = int(orig_height * popup_w / orig_width)  # 従来の高さ
        self.bg_height = previous_height
        self.bg_width  = max(1, popup_w // 3)  # 横だけ縮小
        self.bg_image = self.bg_image_orig.resize(
            (self.bg_width, self.bg_height),
            Image.Resampling.LANCZOS
        )
        self.photo = ImageTk.PhotoImage(self.bg_image)

        # 位置（初期は中央上）
        self.bg_x = (self.popup_width - self.bg_width) // 2
        self.bg_y = 0
        self.base_top_y = 0  # 「上」ポジションの基準Y（↓↑操作で一緒に動く）

        # 左/中央/右 のx座標
        self.x_positions = [
            0 + X_OFFSET,  # 左
            (self.popup_width - self.bg_width) // 2,  # 中央
            self.popup_width - self.bg_width - X_OFFSET  # 右
        ]

        # 直前の位置（5択のインデックス）※連続同一回避用
        self.last_pos_index = None

        # Canvas に背景画像を配置
        self.canvas = tk.Canvas(self.root, width=popup_w, height=popup_h, bg="black")
        self.canvas.pack()
        self.bg_id = self.canvas.create_image(self.bg_x, self.bg_y, image=self.photo, anchor=tk.NW)

        # テキスト
        self.font_size = FONT_SIZE
        self.text_font = tkfont.Font(family="Arial", size=self.font_size, weight="bold")
        self.text_id = self.canvas.create_text(
            self.bg_x + self.bg_width // 2,
            self.bg_y + self.bg_height // 2,
            text="",
            font=self.text_font,
            fill="black",
            width=self.bg_width - 100
        )

        # 自動隠蔽タイマー開始
        self.last_time = time.time()
        self._poll()

    # 便利: clamp
    def _clamp_y(self, y: int) -> int:
        return max(0, min(self.popup_height - self.bg_height, y))

    # 座標セット
    def set_bubble_pos(self, x: int, y: int):
        y = self._clamp_y(y)
        old_x, old_y = self.bg_x, self.bg_y
        self.bg_x, self.bg_y = x, y
        self.canvas.coords(self.bg_id, self.bg_x, self.bg_y)
        self.canvas.coords(
            self.text_id,
            self.bg_x + self.bg_width // 2,
            self.bg_y + self.bg_height // 2
        )
        logger.info(f"バブル位置 (X,Y): ({old_x},{old_y}) → ({self.bg_x},{self.bg_y})")

    # ランダム位置（左上, 中央上, 右上, 左下, 右下）へ移動（連続同位置は避ける）
    def randomize_bubble_pos(self):
        top_y = self._clamp_y(self.base_top_y + Y_OFFSET)
        bottom_y = self._clamp_y(self.base_top_y + BOTTOM_OFFSET)

        candidates = [
            (self.x_positions[0], top_y),    # 0: 左上
            (self.x_positions[1], top_y),    # 1: 中央上
            (self.x_positions[2], top_y),    # 2: 右上
            (self.x_positions[0], bottom_y), # 3: 左下
            (self.x_positions[2], bottom_y), # 4: 右下
        ]

        idxs = list(range(len(candidates)))
        if self.last_pos_index is not None and self.last_pos_index in idxs:
            idxs.remove(self.last_pos_index)

        idx = random.choice(idxs)
        self.last_pos_index = idx
        x, y = candidates[idx]
        self.set_bubble_pos(x, y)

    def _poll(self):
        if time.time() - self.last_time >= LIMIT_TIME:
            self.canvas.itemconfig(self.bg_id, state='hidden')
            self.canvas.itemconfig(self.text_id, state='hidden')
        else:
            self.canvas.itemconfig(self.bg_id, state='normal')
            self.canvas.itemconfig(self.text_id, state='normal')
        self.root.after(500, self._poll)

    def update_text(self, text: str):
        self.last_time = time.time()
        self.canvas.itemconfig(self.bg_id,   state='normal')
        self.canvas.itemconfig(self.text_id, state='normal')

        max_w = self.bg_width - 100
        if self.text_font.measure(text) > max_w:
            for i in range(len(text)):
                sub = text[i:]
                if self.text_font.measure(sub) <= max_w:
                    text = sub
                    break

        self.canvas.itemconfig(self.text_id, text=text)
        self.root.update_idletasks()

    def change_font_size(self, delta):
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
        背景画像とテキストを delta_y 分上下に移動。
        基準Y(base_top_y)も一緒に動かして、下側ポジションが常に「基準+200」に保たれるようにする。
        """
        new_y = self._clamp_y(self.bg_y + delta_y)
        moved = new_y - self.bg_y
        self.base_top_y = self._clamp_y(self.base_top_y + moved)  # 基準も同じだけ動かす
        self.set_bubble_pos(self.bg_x, new_y)

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
    def __init__(self, bubble):
        super().__init__()
        self.gaze_data_list = []
        self.running = False
        self.start_time = None
        self.bubble = bubble
        self.last_trigger = 0.0  # クールダウン管理

        found = tr.find_all_eyetrackers()
        if not found:
            raise RuntimeError("アイトラッカーが見つかりません")
        self.eyetracker = found[0]

    def gaze_callback(self, gaze_data):
        self.gaze_data_list.append(gaze_data)

        # yが 0.7〜0.9 の範囲でトリガー
        y_values = []
        for key in ('left_gaze_point_on_display_area', 'right_gaze_point_on_display_area'):
            pt = gaze_data.get(key)
            if pt and len(pt) == 2:
                y = pt[1]
                if isinstance(y, (int, float)) and 0.0 <= y <= 1.0:
                    y_values.append(y)

        if y_values:
            current_y_min = min(y_values)
            if 0.7 <= current_y_min <= 0.9:
                now = time.time()
                if now - self.last_trigger >= 0.8:  # クールダウン
                    self.last_trigger = now
                    # 初回単語表示時にタイマー開始
                    if self.start_time is None:
                        self.start_time = now
                        logger.info(f"Tobii: タイマー開始（初回単語表示）")
                    text = random_sloan(4)
                    # Tkのメインスレッドで更新
                    self.bubble.root.after(0, self.bubble.update_text, text)
                    self.bubble.root.after(0, self.bubble.randomize_bubble_pos)
                    logger.debug(f"視線y∈[0.7,0.9]→テキスト:{text} & 位置5択ランダム（直前と同一を除外）")

    def start_streaming(self):
        if self.running:
            logger.info("Tobii: すでに計測中です")
            return
        self.running = True
        self.start_time = None  # 初回単語表示時に開始するため、ここではNone
        self.gaze_data_list = []
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

        filename = datetime.now().strftime(
            f"visibility_expt_%Y%m%d_%H%M%S_font{self.bubble.font_size}_y{self.bubble.bg_y}.csv"
        )

        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['left_x','left_y','right_x','right_y'])
            for d in self.gaze_data_list:
                lx, ly = d.get('left_gaze_point_on_display_area', (None, None))
                rx, ry = d.get('right_gaze_point_on_display_area', (None, None))
                writer.writerow([lx, ly, rx, ry])
            # 計測時間を最終行に追加
            writer.writerow(['計測時間', f'{elapsed:.2f}秒'])
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

    # キー操作
    bubble.root.bind_all("a", lambda e: bubble.change_font_size(+DELTA))
    bubble.root.bind_all("s", lambda e: bubble.change_font_size(-DELTA))
    bubble.root.bind_all("<Up>",   lambda e: bubble.move_bubble(-50))
    bubble.root.bind_all("<Down>", lambda e: bubble.move_bubble(+50))

    # 計測トグル
    bubble.root.bind_all("f", lambda e: (tobii_thread.toggle_streaming(), "break"))

    bubble.root.mainloop()

if __name__ == "__main__":
    main()
