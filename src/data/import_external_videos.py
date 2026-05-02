import argparse
import json
import time
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Set
import cv2
import yaml
import sys
import os
import contextlib

# 1. Tắt log ở mức hệ thống (TensorFlow & MediaPipe C++)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '2'
os.environ['ABSL_LOGGING_MIN_LOG_LEVEL'] = '3'

# Import các công cụ chuẩn từ project
from src.utils.hand_detector import HandDetector

# Load cấu hình
with open("configs/data.yaml", encoding="utf-8") as f:
    cfg_data = yaml.safe_load(f)

DATA_CFG = cfg_data["data"]
RECORD_CFG = cfg_data["record"]
RAW_DIR = Path(DATA_CFG["raw_data_dir"])

with open("configs/utils.yaml", encoding="utf-8") as f:
    cfg_utils = yaml.safe_load(f)


def load_video_labels(splits_dir: Path) -> Dict[str, str]:
    """Đọc các file CSV trong splits/ để lấy nhãn (Gloss) của từng video."""
    video_to_label = {}
    csv_files = list(splits_dir.glob("*.csv"))
    
    for csv_path in csv_files:
        print(f"  [>] Đang nạp metadata từ: {csv_path.name}")
        try:
            df = pd.read_csv(csv_path)
            # Map 'Video file' -> 'Gloss'
            for _, row in df.iterrows():
                v_file = row['Video file']
                gloss = row['Gloss']
                video_to_label[v_file] = str(gloss).upper()
        except Exception as e:
            print(f"  [!] Lỗi khi đọc file {csv_path.name}: {e}")
            
    return video_to_label

def get_processed_videos(raw_dir: Path) -> Set[str]:
    processed_stems = set()
    jsonl_files = sorted(raw_dir.glob("session_*.jsonl"))

    if not jsonl_files:
        return processed_stems

    print(f"  [>] Đang quét {len(jsonl_files)} file session cũ trong {raw_dir}...")

    for jsonl_path in jsonl_files:
        try:
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        sample = json.loads(line)
                        parts = sample["sample_id"].split("_")
                        if len(parts) >= 3:
                            video_stem = "_".join(parts[2:])
                            processed_stems.add(video_stem)
        except Exception as e:
            print(f"  [!] Lỗi khi quét file {jsonl_path.name}: {e}")

    return processed_stems

def process_video(video_path: Path, detector: HandDetector, record_fps: float) -> List[List[Dict]]:
    """Trích xuất landmarks thô từ từng frame của video."""
    cap = cv2.VideoCapture(str(video_path))
    sequence_data = []
    frame_duration_ms = 1000.0 / record_fps
    frame_count = 0

    with suppress_native_stderr():
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            custom_timestamp_ms = int(frame_count * frame_duration_ms)

            try:
                result = detector.detect(frame, timestamp_ms=custom_timestamp_ms)
                hands = detector.get_hands_data(result, frame.shape)
                sequence_data.append(hands)
            except Exception as e:
                print(f"[frame {frame_count}] detect lỗi: {e}")
                sequence_data.append([])

            frame_count += 1

    cap.release()
    return sequence_data

def main(override_args=None):
    parser = argparse.ArgumentParser()
    # Đường dẫn lấy từ data.yaml và
    base_data_path = Path(DATA_CFG["external_data_dir"]) 
    parser.add_argument("--videos-dir", type=Path, default=base_data_path / "videos")
    parser.add_argument("--splits-dir", type=Path, default=base_data_path / "splits")
    parser.add_argument("--model-path", type=str, default=cfg_utils["hand_detector"]["model_path"])
    
    args = parser.parse_args(override_args)

    # 1. Nạp từ điển nhãn
    video_labels = load_video_labels(args.splits_dir)

    all_video_files = sorted([
        f for f in args.videos_dir.glob("*") 
        if f.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"]
    ])
    
    if not all_video_files:
        print(f"  [!] Không tìm thấy video nào tại {args.videos_dir}")
        return

    session_id = int(time.time() * 1000)
    output_path = RAW_DIR / f"session_{session_id}.jsonl"
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    processed_stems = get_processed_videos(RAW_DIR)
    video_files_to_process = [v for v in all_video_files if v.stem not in processed_stems]

    print(f"  [>] Tổng cộng: {len(all_video_files)} video.")
    print(f"  [✓] Đã hoàn thành trước đó: {len(processed_stems)} video.")
    print(f"  [⚡] Cần xử lý tiếp: {len(video_files_to_process)} video.")

    if not video_files_to_process:
        print("  [🎉] Đã hoàn thành toàn bộ dataset!")
        return

    imported_count = 0
    fps = float(RECORD_CFG["record_fps"])

    # 5. Vòng lặp xử lý (Chế độ "a" - Append)
    with open(output_path, "a", encoding="utf-8") as f_out:
        for vp in video_files_to_process:
            label = video_labels.get(vp.name)
            if not label:
                # Nếu CSV không có, có thể dự phòng bằng cách tách tên file (như 66221-CARRY.mp4)
                if "-" in vp.stem:
                    label = vp.stem.split("-")[-1].upper()
                else:
                    print(f"    [?] Bỏ qua {vp.name}: Không xác định được nhãn.")
                    continue

            detector = None
            try:
                with suppress_native_stderr():
                    detector = HandDetector(model_path=args.model_path)

                sequence = process_video(vp, detector, fps)

                if sequence:
                    sample = {
                        "sample_id": f"ext_{int(time.time()*1000)}_{vp.stem}",
                        "label": label,
                        "data": sequence
                    }
                    f_out.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    f_out.flush()

                    imported_count += 1
                    if imported_count % 10 == 0 or imported_count == 1:
                        print(f"    [PROGRESS] Đã xong {len(processed_stems) + imported_count}/{len(all_video_files)}...")
            except Exception as e:
                print(f"    [!] Lỗi khi xử lý {vp.name}: {e}")
            finally:
                if detector is not None:
                    try:
                        with suppress_native_stderr():
                            detector.close()
                    except Exception:
                        pass

    print(f"\n[XONG] Đã xử lý thêm {imported_count} mẫu mới.")

@contextlib.contextmanager
def suppress_native_stderr():
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    old_stderr_fd = os.dup(2)
    try:
        os.dup2(devnull_fd, 2)
        yield
    finally:
        os.dup2(old_stderr_fd, 2)
        os.close(old_stderr_fd)
        os.close(devnull_fd)