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

## 参考リンク

https://qiita.com/fiftystorm36/items/b2fd47cf32c7694adc2e

## プロジェクトの構成

```
.
├── project/
│   ├── .venv/                              // 仮想環境
│   ├── images/                             // 表示する写真
│   ├── src/
│   │   ├── detect_faces_gif.py             // 表情認識で分析された感情を漫符として表示（gif画像）
│   │   ├── detect_faces_movie_pyqt.py      // 表情認識で分析された感情を漫符として表示（動画・PyQt使用）
│   │   ├── detect_faces_movie.py           // 表情認識で分析された感情を漫符として表示（動画）
│   │   ├── detect_faces.py                 // 表情認識で分析された感情を漫符として表示（png画像）
│   │   ├── display_translated_scripts.py   // 翻訳したテキストを表示（プロジェクター投影用）
│   │   ├── sample.py                       // 特定のキーボード押下でそれに対応した漫符を表示
│   │   ├── subtitle.py                     // キーボードで入力した内容を漫符で表示した吹き出しの中に表示
│   │   ├── texts_amd_faces_gif.py          // テキスト＋表情で感情分析した結果を漫符に表示（gif画像）
│   │   ├── texts_amd_faces_movie.py        // テキスト＋表情で感情分析した結果を漫符に表示（動画・PyQt使用）
│   │   ├── texts_amd_faces.py              // テキスト＋表情で感情分析した結果を漫符に表示
│   │   ├── transcribe.py                   // テキスト認識で分析された感情を漫符として表示
│   │   ├── translated_speech_bubble.py     // 吹き出しの中に翻訳したテキストを表示
│   │   └── translated_speech_bubble2.py     // 吹き出しの中に翻訳したテキストを表示 + 顔の位置によって吹き出し位置（y軸）が動く
│   └── requirements.txt                    // 仮想環境に必要なライブラリ管理
├── .gitignore
└── README.md
```

## デモ時

**音声ミュート（表情認識のみさせる場合）**

システム設定アプリ > サウンド > 「出力と入力」でマイクの入力音量を最小にする
