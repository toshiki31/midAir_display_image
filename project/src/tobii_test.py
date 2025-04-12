# .tobiienv内で実行してください

import tobii_research as tr

# アイトラッカー検出
eye_trackers = tr.find_all_eyetrackers()
eye_tracker = eye_trackers[0]

# アイトラッカー情報の取得
print("Address: " + eye_tracker.address)
print("Model: " + eye_tracker.model)
print("Name (It's OK if this is empty): " + eye_tracker.device_name)
print("Serial number: " + eye_tracker.serial_number)