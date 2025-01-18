## 利用手順

**Activate(Linux, Mac)**

```bash
source [venvname]/bin/activate
```

**パッケージのインストール**

```bash
([venvname])$ pip install [package name]
```

**Deactivate**

```bash
([venvname])$ deactivate
```

**実行**

```bash
python scr/[filename].py
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
│   ├── .venv/                   // 仮想環境（detect_faces.py, subtitle.py, texts_and_faces.py, transcribe.py）
│   ├── images/                  // 表示する写真
│   ├── sampleEnv/               // 仮想環境（sample.py）
│   ├── src/
│   │   ├── detect_faces.txt     // 表情認識で分析された感情を漫符として表示
│   │   ├── sample.py            // 特定のキーボード押下でそれに対応した漫符を表示
│   │   ├── subtitle.py          // キーボードで入力した内容を漫符で表示した吹き出しの中に表示
│   │   ├── texts_amd_faces.py   // テキスト＋表情で感情分析した結果を漫符に表示
│   │   └── transcribe.py        // テキスト認識で分析された感情を漫符として表示
│   └── requirements.txt        // 仮想環境に必要なライブラリ管理（text_and_faces.py 用）
├── .gitignore
└── README.md
```
