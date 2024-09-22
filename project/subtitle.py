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
popup_label = tk.Label(popup, text="ここに文字が表示されます", font=("Arial", 60))
# ラベルをx=100, y=50の位置に配置
popup_label.place(x=100, y=50)

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
        popup_label.config(text=current_text)
        # 1秒後に次の単語を表示
        popup.after(2000, display_text_slowly, words, index + 1)
    else:
        # テキストを空にする
        popup_label.config(text="")
        # 入力フィールドのテキストを空にする
        entry.delete(0, tk.END)
        # すべての単語を表示し終わったら、入力フィールドにフォーカスを戻す
        set_focus()
        

# 入力内容を別ウィンドウのラベルに反映する関数
def update_label(event=None):
    typed_text = entry.get()  # 入力フィールドからテキストを取得
    words = typed_text.split()  # スペースで分割
    popup_label.config(text="")  # 別ウィンドウのラベルにテキストを表示
    display_text_slowly(words)  # テキストを遅らせて表示

# エンターキーが押されたときにラベルを更新
entry.bind("<Return>", update_label)

# メインループを開始
root.mainloop()
