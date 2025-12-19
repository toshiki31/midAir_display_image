# Pupil Core Face Gaze Recorder

このモジュールは、Pupil Capture の Network API に接続して
gaze と world フレームを受信し、顔検出と注視判定を行い、
MP4 と CSV ログを出力するバックグラウンド録画ツールです。

## 必要なもの

- Pupil Core 本体
- Pupil Capture アプリ（インストール済み）
- Pupil Capture で Network API プラグインを有効化
- 視線キャリブレーション完了（Gaze Mapper）

## Pupil Capture の起動

macOS では通常 /Applications にインストールされます。
Finder から起動するか、Terminal で以下を実行します。

```
sudo /Applications/Pupil\ Capture.app/Contents/MacOS/pupil_capture
```

注意: 通常は sudo 不要です。パスが違う場合は以下で探します。

ls /Applications | rg -i "pupil"
ls ~/Applications | rg -i "pupil"

## Network API の有効化

1. Pupil Capture を開く
2. プラグインメニューから Network API を有効化
3. Remote ポート（デフォルト: 50020）を確認

## キャリブレーション

1. Pupil Capture でキャリブレーションを実行
2. Gaze Mapper が有効になっていることを確認
3. confidence が安定していることを確認

## レコーダーの実行

プロジェクトルートから実行:

python project/src/pupilcore_face_gaze_recorder.py

主なオプション:

--duration 10
--fps 30
--output ./output
--aws
--host 127.0.0.1 --port 50020

## 終了方法

- ターミナルで q または Esc を押す
- または --duration を指定して自動終了

## スクリーンショット/画面手順の追記

必要であれば、以下のように追記できます。

- Network API プラグインの位置が分かるスクリーンショット
- キャリブレーション画面（Gaze Mapper）のスクリーンショット
- 重要な設定項目（Remote ポートなど）のスクリーンショット

## トラブルシューティング

- "Pupil Remote did not respond"
  Pupil Capture が起動していない/Network API 無効/ホストやポート不一致
- "Waiting for world frames"
  world ストリームが動いていない
- "Waiting for gaze data"
  キャリブレーション未完了/ Gaze Mapper が無効
