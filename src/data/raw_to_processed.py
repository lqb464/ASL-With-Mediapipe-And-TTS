import argparse
import json
import math
from pathlib import Path

import numpy as np
import yaml


with open("configs/data.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

DATA_CFG = cfg["data"]
LANDMARKS_CFG = cfg["landmarks"]
LABEL_CFG = cfg["label"]
PROC_CFG = cfg["processing"]

DEFAULT_INPUT = Path(DATA_CFG["raw_data_dir"])
DEFAULT_OUTPUT = Path(DATA_CFG["processed_data_dir"]) / PROC_CFG["default_output_name"]

WRIST = int(LANDMARKS_CFG["wrist"])
NUM_LANDMARKS = int(LANDMARKS_CFG["num_landmarks"])
EPS = float(LABEL_CFG["eps"])

HAND_FEATURE_DIM = NUM_LANDMARKS * 3
FRAME_FEATURE_DIM = HAND_FEATURE_DIM * 2


def get_xyz(p):
    if isinstance(p, dict):
        return float(p.get("x", 0.0)), float(p.get("y", 0.0)), float(p.get("z", 0.0))

    if isinstance(p, (list, tuple)):
        return (
            float(p[0]) if len(p) > 0 else 0.0,
            float(p[1]) if len(p) > 1 else 0.0,
            float(p[2]) if len(p) > 2 else 0.0,
        )

    return 0.0, 0.0, 0.0


def normalize_landmarks(hand_data: dict):
    if not hand_data or "landmarks" not in hand_data:
        return None

    lmks = hand_data["landmarks"]
    if not lmks or len(lmks) != NUM_LANDMARKS:
        return None

    wrist_x, wrist_y, wrist_z = get_xyz(lmks[WRIST])

    translated = []
    for p in lmks:
        x, y, z = get_xyz(p)
        translated.append((x - wrist_x, y - wrist_y, z - wrist_z))

    max_dist = 0.0
    for x, y, z in translated:
        dist = math.sqrt(x * x + y * y + z * z)
        max_dist = max(max_dist, dist)

    scale = 1.0 / (max_dist + EPS)

    flat = []
    for x, y, z in translated:
        flat.extend([x * scale, y * scale, z * scale])

    return flat


def get_handedness(hand: dict):
    return str(hand.get("handedness", hand.get("label", ""))).lower()


def extract_features_from_raw(raw_sample: dict, score_threshold: float):
    sequence_features = []

    for frame_hands in raw_sample["data"]:
        left = [0.0] * HAND_FEATURE_DIM
        right = [0.0] * HAND_FEATURE_DIM
        has_valid_hand = False
        unknown_slot = 0

        for hand in frame_hands:
            score = hand.get("score", 0.0)
            score = 0.0 if score is None else float(score)

            if score < score_threshold:
                continue

            flat = normalize_landmarks(hand)
            if flat is None:
                continue

            handedness = get_handedness(hand)

            if handedness == "left":
                left = flat
            elif handedness == "right":
                right = flat
            else:
                if unknown_slot == 0:
                    left = flat
                else:
                    right = flat
                unknown_slot += 1

            has_valid_hand = True

        if has_valid_hand:
            sequence_features.append(left + right)

    if not sequence_features:
        return None

    return np.array(sequence_features, dtype=np.float32)


def pad_or_truncate_into(seq: np.ndarray, target_len: int, out_row: np.ndarray):
    out_row.fill(0.0)

    n = min(len(seq), target_len)
    if n > 0:
        out_row[:n, :] = seq[:n, :]


def get_config_target_len():
    value = PROC_CFG.get(
        "sequence_length",
        PROC_CFG.get("max_sequence_length", PROC_CFG.get("max_frames", None))
    )
    return int(value) if value is not None else None


def scan_raw_files(input_files, score_threshold):
    labels = []
    sample_ids = []
    max_len = 0
    dropped = 0

    print("[PASS 1] Scanning RAW metadata...")

    for f_path in input_files:
        print(f"[>] Scan: {f_path.name}")

        with f_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                raw_sample = json.loads(line)
                feat = extract_features_from_raw(raw_sample, score_threshold)

                if feat is None:
                    dropped += 1
                    continue

                labels.append(raw_sample["label"])
                sample_ids.append(raw_sample["sample_id"])
                max_len = max(max_len, len(feat))

    return labels, sample_ids, max_len, dropped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--score-threshold", type=float, default=float(LABEL_CFG["threshold_score"]))
    parser.add_argument("--sequence-length", type=int, default=None)
    parser.add_argument("--compressed", action="store_true", help="Lưu npz compressed. Chậm hơn và có thể tốn RAM hơn.")
    args = parser.parse_args()

    input_files = sorted(args.input.glob("*.jsonl"))
    if not input_files:
        print(f"Không tìm thấy file .jsonl nào tại {args.input}")
        return

    labels_raw, sample_ids, max_len, dropped_count = scan_raw_files(input_files, args.score_threshold)

    if not labels_raw:
        print("Không có dữ liệu hợp lệ để xử lý.")
        return

    target_len = args.sequence_length or get_config_target_len() or max_len

    unique_labels = sorted(set(labels_raw))
    label_to_id = {label: i for i, label in enumerate(unique_labels)}
    y = np.array([label_to_id[label] for label in labels_raw], dtype=np.int64)
    sample_ids = np.array(sample_ids)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    tmp_x_path = args.output.with_suffix(".X.tmp.npy")
    print("[PASS 2] Writing X by memmap, không giữ toàn bộ RAM...")
    print(f"[i] Target shape: ({len(labels_raw)}, {target_len}, {FRAME_FEATURE_DIM})")

    X_mm = np.lib.format.open_memmap(
        tmp_x_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(labels_raw), target_len, FRAME_FEATURE_DIM),
    )

    row_idx = 0

    for f_path in input_files:
        print(f"[>] Process: {f_path.name}")

        with f_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                raw_sample = json.loads(line)
                feat = extract_features_from_raw(raw_sample, args.score_threshold)

                if feat is None:
                    continue

                pad_or_truncate_into(feat, target_len, X_mm[row_idx])
                row_idx += 1

    X_mm.flush()

    X_read = np.load(tmp_x_path, mmap_mode="r")

    if args.compressed:
        np.savez_compressed(
            args.output,
            X=X_read,
            y=y,
            sample_ids=sample_ids,
            label_map=json.dumps(label_to_id, ensure_ascii=False),
        )
    else:
        np.savez(
            args.output,
            X=X_read,
            y=y,
            sample_ids=sample_ids,
            label_map=json.dumps(label_to_id, ensure_ascii=False),
        )

    tmp_x_path.unlink(missing_ok=True)

    meta_path = args.output.with_name("train_meta.json")

    meta = {
        "label_map": label_to_id,          # dict label -> id
        "max_len": int(target_len),        # sequence length
        "num_classes": len(label_to_id),
        "feature_dim": FRAME_FEATURE_DIM,
    }

    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[✓] Đã lưu meta tại: {meta_path}")

    print(f"[✓] Đã lưu dataset processed tại: {args.output}")
    print(f"[✓] Shape X: {X_read.shape}")
    print(f"[✓] Feature/frame: {FRAME_FEATURE_DIM} = left({HAND_FEATURE_DIM}) + right({HAND_FEATURE_DIM})")
    print(f"[✓] Tổng mẫu giữ lại: {len(labels_raw)}")
    print(f"[i] Mẫu bị loại: {dropped_count}")