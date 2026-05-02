import json
import time
from pathlib import Path
import cv2
import yaml

# Load cấu hình từ data.yaml
with open("configs/data.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

with open("configs/utils.yaml", encoding="utf-8") as f:
    label_cfg = yaml.safe_load(f)

DATA_CFG = cfg["data"]
LABEL_CFG = cfg["label"]

RAW_DIR = Path(DATA_CFG["raw_data_dir"])
SILENCE_LABEL = LABEL_CFG["silence_label"]

def init_labeler():
    """Tạo thư mục raw nếu chưa có."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

def close_labeler():
    pass

def save_session_to_jsonl(all_samples: list) -> Path | None:
    """Lưu toàn bộ clip vào file .jsonl, mỗi clip 1 dòng."""
    if not all_samples:
        return None

    init_labeler()
    session_ts = int(time.time() * 1000)
    session_id = f"session_{session_ts}"
    output_path = RAW_DIR / f"{session_id}.jsonl"

    # Ghi dữ liệu thô: mỗi dòng là một JSON object hoàn chỉnh của 1 sample
    with output_path.open("w", encoding="utf-8") as f:
        for sample in all_samples:
            # separators=(',', ':') loại bỏ khoảng trắng thừa để file gọn nhất
            line = json.dumps(sample, ensure_ascii=False, separators=(',', ':'))
            f.write(line + "\n")

    return output_path

def _draw_text(image, text, org, font_scale=0.65, color=(255, 255, 255), thickness=1):
    cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

def _get_text_size(text, font_scale=0.65, thickness=1):
    (w, h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    return w, h, baseline

def _fit_text_scale(text, max_width, start_scale=0.65, min_scale=0.42, thickness=1):
    scale = start_scale
    while scale >= min_scale:
        w, _, _ = _get_text_size(text, font_scale=scale, thickness=thickness)
        if w <= max_width: return scale
        scale -= 0.02
    return min_scale

def ask_label(preview_frame, num_frames: int | None, window_name="Hand Detection"):
    typed = ""
    while True:
        canvas = preview_frame.copy()
        h, w = canvas.shape[:2]
        blurred = cv2.GaussianBlur(canvas, (15, 15), 0)
        canvas = cv2.addWeighted(canvas, 0.35, blurred, 0.65, 0)
        box_w, box_h = int(w * 0.78), int(h * 0.62)
        x1, y1 = (w - box_w) // 2, (h - box_h) // 2
        x2, y2 = x1 + box_w, y1 + box_h
        overlay = canvas.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (32, 32, 32), -1)
        cv2.addWeighted(overlay, 0.88, canvas, 0.12, 0, canvas)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (110, 110, 110), 1)
        pad_x, current_y = 28, y1 + 42
        max_text_width = box_w - pad_x * 2
        _draw_text(canvas, "ASL LABELING", (x1 + pad_x, current_y), font_scale=0.9, color=(255, 255, 255), thickness=2)
        frame_text = f"Frames: {num_frames}"
        f_w, _, _ = _get_text_size(frame_text, font_scale=0.62)
        _draw_text(canvas, frame_text, (x2 - pad_x - f_w, current_y), font_scale=0.62, color=(210, 210, 210))
        divider_y = current_y + 18
        cv2.line(canvas, (x1 + pad_x, divider_y), (x2 - pad_x, divider_y), (90, 90, 90), 1)
        help_lines = ["press Enter to confirm label (empty=SILENCE)", "skip: skip clip", "ESC: SAVE & EXIT"]
        current_y = divider_y + 34
        for line in help_lines:
            sc = _fit_text_scale(line, max_text_width)
            _draw_text(canvas, line, (x1 + pad_x, current_y), font_scale=sc)
            _, th, _ = _get_text_size(line, font_scale=sc)
            current_y += th + 18
        current_y += 8
        _draw_text(canvas, "Label:", (x1 + pad_x, current_y), font_scale=0.62, color=(220, 220, 220))
        ib_y1, ib_y2 = current_y + 16, current_y + 64
        cv2.rectangle(canvas, (x1 + pad_x, ib_y1), (x2 - pad_x, ib_y2), (70, 70, 70), -1)
        cv2.rectangle(canvas, (x1 + pad_x, ib_y1), (x2 - pad_x, ib_y2), (120, 120, 120), 1)
        blink = (int(time.time() * 2) % 2) == 0
        display_text = typed + ("|" if blink else "")
        t_sc = _fit_text_scale(display_text if display_text else " ", (x2-x1)-pad_x*2-20, 0.7)
        _draw_text(canvas, display_text, (x1 + pad_x + 10, ib_y1 + 32), font_scale=t_sc)
        cv2.imshow(window_name, canvas)
        key = cv2.waitKey(1) & 0xFF
        if key == 13: return typed.strip() if typed.strip() else SILENCE_LABEL
        if key == 27: return 27
        if key in (8, 127): typed = typed[:-1]
        elif 32 <= key <= 126: typed += chr(key)