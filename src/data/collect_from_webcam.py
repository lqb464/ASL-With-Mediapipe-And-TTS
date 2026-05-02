import math
import time
import cv2
import yaml

from src.data.label_data import (
    ask_label,
    close_labeler,
    init_labeler,
    save_session_to_jsonl,
)
from src.utils.hand_detector import HandDetector
from src.utils.webcam import Webcam

# Tải cấu hình
with open("configs/data.yaml", encoding="utf-8") as f:
    data_cfg = yaml.safe_load(f)
with open("configs/utils.yaml", encoding="utf-8") as f:
    utils_cfg = yaml.safe_load(f)

CAM_CFG = utils_cfg["webcam"]
REC_CFG = data_cfg["record"]

WINDOW_NAME = "Hand Detection"
RECORD_FPS = int(REC_CFG["record_fps"])
RECORD_SECONDS = float(REC_CFG["record_seconds"])
COUNTDOWN_SECONDS = float(REC_CFG["countdown_seconds"])
# Tổng số frame cần thu cho 1 clip dựa trên config
TOTAL_TARGET_FRAMES = int(RECORD_SECONDS * RECORD_FPS)

def main():
    cam = Webcam(camera_index=int(CAM_CFG["index"]), width=int(CAM_CFG["width"]), 
                 height=int(CAM_CFG["height"]), fps=int(CAM_CFG["fps"]))
    detector = HandDetector()
    init_labeler()

    all_collected_samples = []
    start_session_time = time.time()
    sequence = []
    countdown_active = True
    recording = False
    countdown_start = time.time()
    last_frame_time = 0.0
    last_label = None

    try:
        while True:
            frame = cam.read()
            if frame is None: break

            now = time.time()
            # MediaPipe Detection
            timestamp_ms = int((now - start_session_time) * 1000)
            result = detector.detect(frame, timestamp_ms=timestamp_ms)
            frame = detector.draw_hands(frame, result)
            hands = detector.get_hands_data(result, frame.shape)

            status = "IDLE"

            # 1. Logic Đếm ngược
            if countdown_active and not recording:
                elapsed = now - countdown_start
                remaining = max(0.0, COUNTDOWN_SECONDS - elapsed)
                status = f"COUNTDOWN: {math.ceil(remaining)}s"
                if elapsed >= COUNTDOWN_SECONDS:
                    countdown_active, recording = False, True
                    sequence = []
                    last_frame_time = 0.0 # Reset trigger cho clip mới

            # 2. Logic Ghi hình (ép đủ số frame)
            elif recording:
                status = f"RECORDING... | Frame: {len(sequence)}/{TOTAL_TARGET_FRAMES}"
                
                # Kiểm tra interval để ghi frame theo RECORD_FPS
                if now - last_frame_time >= (1.0 / RECORD_FPS):
                    sequence.append(hands)
                    last_frame_time = now

                # Kết thúc clip khi đủ số frame mục tiêu
                if len(sequence) >= TOTAL_TARGET_FRAMES:
                    recording = False
                    label_input = ask_label(frame.copy(), len(sequence), WINDOW_NAME)
                    
                    if label_input == 27: # ESC
                        break 
                    elif str(label_input).lower() != "skip":
                        all_collected_samples.append({
                            "sample_id": f"s_{int(time.time()*1000)}",
                            "label": label_input,
                            "data": sequence
                        })
                        last_label = label_input
                    
                    # Reset để quay clip tiếp theo
                    countdown_active, countdown_start = True, time.time()

            # Vẽ UI
            cv2.putText(frame, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Session Samples: {len(all_collected_samples)}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            cv2.imshow(WINDOW_NAME, frame)
            
            if cv2.waitKey(1) & 0xFF == 27: break
    finally:
        if all_collected_samples:
            path = save_session_to_jsonl(all_collected_samples)
            print(f"Lưu thành công {len(all_collected_samples)} clips vào: {path}")
        cam.release()
        cv2.destroyAllWindows()
        close_labeler()