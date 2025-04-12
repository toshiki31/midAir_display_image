import time
import tkinter as tk
import tkinter.font as tkfont
from PIL import Image, ImageTk
import screeninfo

# ===============================
#   定数
# ===============================
SPEECH_BUBBLE1 = "./images/speech-bubble1.png"

# ===============================
#   吹き出し表示クラス (SpeechBubble)
# ===============================
class SpeechBubble:
    def __init__(self):
        self.root = tk.Tk()
        self.root.withdraw()

        self.popup = tk.Toplevel(self.root)
        self.popup.title("Speech Bubble")
        self.popup.attributes("-topmost", True)
        self.popup.update_idletasks()
        self.popup.update()

        monitors = screeninfo.get_monitors()
        if len(monitors) > 1:
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

        aspect_ratio = 16 / 10
        popup_width = int(monitor_height * aspect_ratio)
        popup_height = monitor_height
        self.popup.geometry(f"{popup_width}x{popup_height}+{monitor_x}+{monitor_y}")

        self.popup_width = popup_width
        self.popup_height = popup_height

        bg_image_path = SPEECH_BUBBLE1
        self.bg_image_orig = Image.open(bg_image_path)
        orig_width, orig_height = self.bg_image_orig.size
        new_width = popup_width
        new_height = int(orig_height * new_width / orig_width)
        self.bg_image = self.bg_image_orig.resize((new_width, new_height), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(self.bg_image)

        self.canvas = tk.Canvas(self.popup, width=popup_width, height=popup_height)
        self.canvas.pack()
        self.bg_image_id = self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        self.text_width_limit = popup_width - 100
        self.text_id = self.canvas.create_text(popup_width / 2, new_height // 2,
                                               text="",
                                               font=("Arial", 200, "bold"),
                                               fill="black",
                                               width=self.text_width_limit)
        self.text_font = tkfont.Font(family="Arial", size=200, weight="bold")

    def update_text(self, text):
        if self.text_font.measure(text) <= self.text_width_limit:
            display_text = text
        else:
            display_text = text
            for i in range(len(text)):
                substring = text[i:]
                if self.text_font.measure(substring) <= self.text_width_limit:
                    display_text = substring
                    break
        self.canvas.itemconfig(self.text_id, text=display_text)
        self.root.update_idletasks()

    def show(self):
        self.popup.deiconify()
        self.root.update()

# ===============================
#   メイン処理
# ===============================
def main():
    speech_bubble = SpeechBubble()
    speech_bubble.show()
    speech_bubble.update_text("テキスト")
    speech_bubble.root.mainloop()

if __name__ == "__main__":
    main()
