import tkinter as tk
from PIL import Image, ImageTk
import screeninfo

# メインウィンドウを作成
root = tk.Tk()

# メインウィンドウを最上位に設定
root.attributes("-topmost", True)

# 画面の幅と高さを取得
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# ウィンドウの幅と高さを設定（例えば300x200）
window_width = 300
window_height = 150

# 右下に配置するための位置を計算
x_pos = screen_width - window_width
y_pos = screen_height - window_height

# ウィンドウを右下に表示
root.geometry(f"{window_width}x{window_height}+{x_pos}+{y_pos}")

# 別ウィンドウを作成
popup = tk.Toplevel()
popup.title("表示ウィンドウ")

# スクリーン情報の取得（複数のモニターがある場合）
monitors = screeninfo.get_monitors()

# 別画面があるか確認（最初のモニター以外を使用）
if len(monitors) > 1:
    second_monitor = monitors[1]  # 2番目のモニターを使用
    monitor_width = second_monitor.width
    monitor_height = second_monitor.height
    monitor_x = second_monitor.x
    monitor_y = second_monitor.y
else:
    # 別画面がない場合、メイン画面を使用
    monitor_width = screen_width
    monitor_height = screen_height
    monitor_x = 0
    monitor_y = 0

# 16:10のアスペクト比を考慮したウィンドウサイズを計算
aspect_ratio = 16 / 10
popup_width = int(monitor_height * aspect_ratio)  # 高さに基づいて幅を設定
popup_height = monitor_height  # 高さは画面の高さに合わせる

# 画面全体に16:10のウィンドウを表示（別モニターに）
popup.geometry(f"{popup_width}x{popup_height}+{monitor_x}+{monitor_y}")

# 画像の読み込み
img = Image.open("images/fukidashi.png")
img = img.resize((popup_width, 200), Image.Resampling.LANCZOS)
photo = ImageTk.PhotoImage(img)

# Canvasを作成して画像を表示
canvas = tk.Canvas(popup, width=popup_width, height=popup_height)
canvas.pack()
canvas.create_image(0, 0, image=photo, anchor=tk.NW)

# テキストを画像内に表示する
text_id = canvas.create_text(popup_width / 2, 80, text="ここに文字が表示されます", font=("Arial", 60), fill="black")

# テキスト入力フィールド
entry = tk.Entry(root, width=30)
entry.pack(pady=20)

# 初期フォーカスをテキスト入力フィールドに設定
def set_focus():
    entry.focus_set()

# ウィンドウが開かれてから少し後にフォーカスを設定
root.after(100, set_focus)

# テキスト表示を遅らせるための関数
def display_text_slowly(words, index=0):
    if index < len(words):
        # これまでのテキストに現在の単語を追加して表示
        current_text = words[index]
        canvas.itemconfig(text_id, text=current_text)
        # 1秒後に次の単語を表示
        popup.after(2000, display_text_slowly, words, index + 1)
    else:
        # テキストを空にする
        canvas.itemconfig(text_id, text="")
        # 入力フィールドのテキストを空にする
        entry.delete(0, tk.END)
        # すべての単語を表示し終わったら、入力フィールドにフォーカスを戻す
        set_focus()

# 入力内容を別ウィンドウのラベルに反映する関数
def update_label(event=None):
    typed_text = entry.get()  # 入力フィールドからテキストを取得
    words = typed_text.split()  # スペースで分割
    canvas.itemconfig(text_id, text="")  # 画像内にテキストを表示
    display_text_slowly(words)  # テキストを遅らせて表示

# エンターキーが押されたときにラベルを更新
entry.bind("<Return>", update_label)

# メインループを開始
root.mainloop()
