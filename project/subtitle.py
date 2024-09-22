import tkinter as tk

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
size = root.maxsize()
popup.geometry('{}x{}+0+0'.format(*size))


# 別ウィンドウに表示するためのラベルを作成
popup_label = tk.Label(popup, text="ここに文字が表示されます", font=("Arial", 20))
popup_label.pack(pady=20)

# テキスト入力フィールド
entry = tk.Entry(root, width=30)
entry.pack(pady=20)

# 初期フォーカスをテキスト入力フィールドに設定
def set_focus():
    entry.focus_set()

# ウィンドウが開かれてから少し後にフォーカスを設定
root.after(100, set_focus)

# 入力内容を別ウィンドウのラベルに反映する関数
def update_label(event=None):
    typed_text = entry.get()  # 入力フィールドからテキストを取得
    popup_label.config(text=typed_text)  # 別ウィンドウのラベルにテキストを表示
# エンターキーが押されたときにラベルを更新
entry.bind("<Return>", update_label)

# メインループを開始
root.mainloop()
