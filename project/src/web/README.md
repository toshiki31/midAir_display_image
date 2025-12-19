# Web版 吹き出し表示システム

iPhoneやiPadなどのモバイルデバイスのブラウザで、リアルタイムに翻訳された吹き出しを表示するシステムです。

## 📱 概要

このシステムは以下の機能を提供します：

- **リアルタイム音声認識**: AWS Transcribeを使用した音声のテキスト化
- **自動翻訳**: AWS Translateによる多言語翻訳
- **顔検出**: OpenCVによる顔検出に基づく吹き出し位置の自動調整
- **WebSocket通信**: iPhoneなどのブラウザにリアルタイムで吹き出しを表示
- **無音検出**: 発話がない時は自動的に吹き出しを非表示

## 🚀 セットアップ

### 1. 依存関係のインストール

```bash
cd project
pip install -r requirements.txt
```

必要なパッケージ:
- Flask (Webサーバー)
- Flask-SocketIO (WebSocket通信)
- その他、既存の音声認識・画像処理ライブラリ

### 2. AWS認証情報の設定

AWS Transcribe、Translate、Comprehendを使用するため、AWS認証情報が必要です。

```bash
aws configure
```

または環境変数で設定:
```bash
export AWS_ACCESS_KEY_ID="your_access_key"
export AWS_SECRET_ACCESS_KEY="your_secret_key"
export AWS_DEFAULT_REGION="ap-northeast-1"
```

## 📖 使い方

### 1. サーバーの起動

```bash
cd project/src/web
python translated_speech_bubble_web.py
```

起動すると、言語選択ダイアログが表示されます：
- **翻訳元言語**: 話す言語を選択（例: ja-JP）
- **翻訳先言語**: 表示する言語を選択（例: en-US）

起動後、以下のようなメッセージが表示されます：

```
============================================================
🎉 Server is running!
============================================================

📱 iPhoneのブラウザで以下のURLにアクセスしてください:

   http://192.168.1.5:5000

   (または http://localhost:5000 でローカル確認)

============================================================
```

### 2. iPhoneでアクセス

1. iPhoneをMacと**同じWiFiネットワーク**に接続
2. SafariまたはChromeを開く
3. 表示されたURL（例: `http://192.168.1.5:5000`）にアクセス
4. 画面に「接続済み」と表示されれば成功

### 3. 使用開始

- マイクに向かって話すと、自動的に音声認識→翻訳が実行されます
- 翻訳された吹き出しがiPhone画面に表示されます
- カメラで顔を検出すると、顔の位置に応じて吹き出しの位置が調整されます
- 1秒間無音が続くと、吹き出しは自動的に非表示になります

## 🔧 設定のカスタマイズ

`translated_speech_bubble_web.py`の以下の定数で動作を調整できます：

```python
SILENCE_THRESHOLD = 1.0  # 無音検出の閾値（秒）
SAMPLE_RATE = 16000      # 音声サンプリングレート
CHUNK_SIZE = 1024        # 音声チャンクサイズ
```

顔検出のオフセット調整（吹き出しを顔から何ピクセル離すか）:

```python
face_thread = FaceDetectionThread(offset=600)  # offsetの値を調整
```

## 📂 ファイル構成

```
project/src/web/
├── translated_speech_bubble_web.py  # メインアプリケーション
├── templates/
│   └── speech_bubble.html           # iPhone表示用HTML
├── static/
│   └── images/
│       ├── speech-bubble1.png       # 吹き出し画像1
│       ├── speech-bubble2.png       # 吹き出し画像2
│       └── speech-bubble3.png       # 吹き出し画像3
└── README.md                         # このファイル
```

## 🌐 複数デバイスでの同時表示

このシステムは複数のデバイスで同時に表示できます：

1. 複数のiPhone/iPadで同じURLにアクセス
2. すべてのデバイスに同じ吹き出しが表示されます
3. 実験や発表に便利

## 🐛 トラブルシューティング

### iPhoneから接続できない

- MacとiPhoneが**同じWiFiネットワーク**にあることを確認
- Macのファイアウォールでポート5000が許可されているか確認
- URLのIPアドレスが正しいか確認（`ifconfig`で確認）

### 音声が認識されない

- マイクの入力音量を確認（システム環境設定 > サウンド）
- マイクへのアクセス許可を確認
- AWS認証情報が正しく設定されているか確認

### 吹き出しの位置がおかしい

- `FaceDetectionThread(offset=600)`の`offset`値を調整
- カメラが正しく動作しているか確認

### 翻訳されない

- AWS Translateの利用権限を確認
- ソース言語とターゲット言語が正しく設定されているか確認
- AWS認証情報のリージョンが`ap-northeast-1`になっているか確認

## 💡 Tips

- **全画面表示**: iPhoneのSafariで「共有」→「ホーム画面に追加」でアプリ風に使用可能
- **省電力**: 長時間使用する場合はiPhoneを充電しながら使用推奨
- **ネットワーク**: 安定したWiFi環境で使用することを推奨（レイテンシ50-150ms）

## 📊 システム要件

- **Mac**: macOS 10.15以降、Webカメラ、マイク
- **iPhone/iPad**: iOS 12以降、SafariまたはChrome
- **ネットワーク**: 同一ローカルネットワーク
- **AWS**: Transcribe、Translate、Comprehendのアクセス権限

## 🔗 関連ファイル

元となった実装: `project/src/translated_speech_bubble2.py`

## ライセンス

このプロジェクトは研究・実験用途で作成されています。
