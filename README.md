## 利用手順

**Activate(Linux, Mac)**

```bash
source [venvname]/bin/activate
```

**パッケージのインストール**

```bash
([venvname])$ pip install [package name]
```

**requirements.txt に書き出し**

```bash
pip freeze > requirements.txt
```

**Deactivate**

```bash
([venvname])$ deactivate
```

**実行**

```bash
python src/[filename].py
```

## 新しい環境の作成

```bash
cd [project dir]
python3 -m venv [venvname]
```

## AWS 初期設定

音声認識・翻訳機能を使用するには AWS CLI と認証情報の設定が必要です。

### 1. AWS CLI のインストール

**Homebrew を使用（推奨）**
```bash
brew install awscli
```

**公式インストーラーを使用**
```bash
curl "https://awscli.amazonaws.com/AWSCLIV2.pkg" -o "/tmp/AWSCLIV2.pkg"
sudo installer -pkg /tmp/AWSCLIV2.pkg -target /
```

### 2. AWS 認証情報の設定

**前のPCから認証情報を取得する場合**
```bash
# 前のPCで実行
cat ~/.aws/credentials
cat ~/.aws/config
```

**新しいPCで認証情報ファイルを作成**
```bash
# ディレクトリ作成
mkdir -p ~/.aws

# 認証情報ファイルを作成（前のPCから取得した内容を貼り付け）
nano ~/.aws/credentials
# または直接コピー
cat > ~/.aws/credentials << 'EOF'
[default]
aws_access_key_id = YOUR_ACCESS_KEY
aws_secret_access_key = YOUR_SECRET_KEY

[rekognition]
aws_access_key_id = YOUR_ACCESS_KEY
aws_secret_access_key = YOUR_SECRET_KEY
EOF

# 設定ファイルを作成
cat > ~/.aws/config << 'EOF'
[default]
region = ap-northeast-1
output = json

[profile rekognition]
region = ap-northeast-1
output = json
EOF

# パーミッション設定
chmod 600 ~/.aws/credentials
chmod 600 ~/.aws/config
```

**接続テスト**
```bash
aws sts get-caller-identity --profile rekognition
```

成功すると、ユーザー情報が表示されます。

## セットアップ時のメモ

- `contourpy==1.3.3` は Python 3.11 以上が必要だったため、Python 3.9 の仮想環境では `pip install -r requirements.txt` が失敗する。Homebrew で 3.11 系を入れて仮想環境を作り直し、pip を更新してから依存を入れる。
  ```bash
  brew install python@3.11
  cd /Users/toshiki/Desktop/dev/midAir_display_image/project
  rm -rf .venv
  /opt/homebrew/bin/python3.11 -m venv .venv
  source .venv/bin/activate
  python -m pip install --upgrade pip
  pip install -r requirements.txt
  ```
- `PyAudio` のビルドでは PortAudio のヘッダ (`portaudio.h`) が無いと失敗するので、事前に `brew install portaudio` を実行してから再度 `pip install -r requirements.txt` を行う。
- Web版アプリケーションには Flask と Flask-SocketIO が必要です。`requirements.txt` に含まれていない場合は以下でインストール：
  ```bash
  pip install flask flask-socketio
  ```

## 参考リンク

https://qiita.com/fiftystorm36/items/b2fd47cf32c7694adc2e

## バージョン管理

バージョン管理は pyenv で行っている

**現在のバージョン確認**

```bash
python -V
```

\*\* pyenv で特定のバージョンをインストール

```bash
pyenv install [version]
```

** バージョンを設定（ローカル）**

```bash
pyenv local [version]
```

現在使用しているバージョンが `.python-version`に記載されている

## プロジェクトの構成

```
.
├── project/
│   ├── .venv/                              // 仮想環境
│   ├── images/                             // 表示する写真
│   ├── src/
│   │   ├── web/                            // Web版アプリケーション
│   │   │   ├── translated_speech_bubble_web.py  // Web版：音声認識・翻訳・吹き出し表示
│   │   │   ├── templates/
│   │   │   │   └── speech_bubble.html      // 吹き出し表示用HTMLテンプレート
│   │   │   └── static/
│   │   │       └── images/                 // 吹き出し画像（1行、2行、3行以上用）
│   │   ├── detect_faces_gif.py             // 表情認識で分析された感情を漫符として表示（gif画像）
│   │   ├── detect_faces_movie_pyqt.py      // 表情認識で分析された感情を漫符として表示（動画・PyQt使用）
│   │   ├── detect_faces_movie.py           // 表情認識で分析された感情を漫符として表示（動画）
│   │   ├── detect_faces.py                 // 表情認識で分析された感情を漫符として表示（png画像）
│   │   ├── display_translated_scripts.py   // 翻訳したテキストを表示（プロジェクター投影用）
│   │   ├── display_translated_scripts.py   // 翻訳したテキストを表示+tobiiで視線計測（プロジェクター投影、会話実験用）
│   │   ├── sample.py                       // 特定のキーボード押下でそれに対応した漫符を表示
│   │   ├── subtitle.py                     // キーボードで入力した内容を漫符で表示した吹き出しの中に表示
│   │   ├── texts_amd_faces_gif.py          // テキスト＋表情で感情分析した結果を漫符に表示（gif画像）
│   │   ├── texts_amd_faces_movie.py        // テキスト＋表情で感情分析した結果を漫符に表示（動画・PyQt使用）
│   │   ├── texts_amd_faces.py              // テキスト＋表情で感情分析した結果を漫符に表示
│   │   ├── transcribe.py                   // テキスト認識で分析された感情を漫符として表示
│   │   ├── translated_speech_bubble.py     // 吹き出しの中に翻訳したテキストを表示
│   │   ├── translated_speech_bubble2_tobii.py     // 吹き出しの中に翻訳したテキストを表示 + 顔の位置によって吹き出し位置（y軸）が動く + tobiiで視線計測（会話実験用）
│   │   └── translated_speech_bubble2.py     // 吹き出しの中に翻訳したテキストを表示 + 顔の位置によって吹き出し位置（y軸）が動く
│   └── requirements.txt                    // 仮想環境に必要なライブラリ管理
├── .gitignore
└── README.md
```

## Web版アプリケーションの実行

Web版は音声認識・翻訳・吹き出し表示をブラウザベースで実行できます。

### 1. サーバーの起動

```bash
cd /Users/toshiki/Desktop/dev/midAir_display_image/project
source .venv/bin/activate
python ./src/web/translated_speech_bubble_web.py
```

サーバーが起動すると、アクセス用のURLが表示されます：
```
============================================================
🎉 Server is running!
============================================================

📱 iPhoneのブラウザで以下のURLにアクセスしてください:

   http://192.168.11.6:5010

   (または http://localhost:5010 でローカル確認)

💡 ブラウザで言語を選択してシステムを開始してください
============================================================
```

### 2. ブラウザでアクセス

- **iPhone/スマートフォン**: 表示されたURL（例：`http://192.168.11.6:5010`）にアクセス
- **同じPC**: `http://localhost:5010` にアクセス

### 3. 言語を選択して開始

1. ブラウザに言語選択モーダルが表示されます
2. **翻訳元言語**と**翻訳先言語**を選択
3. 「システムを開始」ボタンをクリック
4. カメラと音声認識が起動し、吹き出し表示が始まります

### 利用可能な言語

- 🇯🇵 日本語 (`ja-JP`)
- 🇺🇸 English (`en-US`)
- 🇨🇳 中文（简体）(`zh-CN`)
- 🇹🇼 中文（繁體）(`zh-TW`)
- 🇰🇷 한국어 (`ko-KR`)
- 🇫🇷 Français (`fr-FR`)
- 🇩🇪 Deutsch (`de-DE`)
- 🇪🇸 Español (`es-ES`)

### コマンドラインオプション

```bash
# デフォルト設定で起動（ポート5010）
python ./src/web/translated_speech_bubble_web.py

# カスタムポートで起動
python ./src/web/translated_speech_bubble_web.py --port 8080

# ホストとポートを指定
python ./src/web/translated_speech_bubble_web.py --host 0.0.0.0 --port 8080

# ヘルプを表示
python ./src/web/translated_speech_bubble_web.py --help
```

### 機能

- ✅ **音声認識**: AWS Transcribeでリアルタイム音声認識
- ✅ **翻訳**: AWS Translateで選択した言語に自動翻訳
- ✅ **吹き出し表示**: 翻訳されたテキストを漫画風の吹き出しで表示
- ✅ **顔検出**: カメラで顔を検出し、吹き出しの位置を自動調整
- ✅ **レスポンシブ**: テキストの長さに応じて吹き出しサイズとフォントサイズを自動調整

## デモ時

**音声ミュート（表情認識のみさせる場合）**

システム設定アプリ > サウンド > 「出力と入力」でマイクの入力音量を最小にする
