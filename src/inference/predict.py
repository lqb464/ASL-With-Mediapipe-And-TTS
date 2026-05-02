from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Deque, Dict, Tuple, List

import numpy as np
import torch
import yaml

# Import logic xử lý từ file raw_to_processed.py để đảm bảo tính nhất quán
from src.data.raw_to_processed import normalize_landmarks, get_handedness, HAND_FEATURE_DIM, FRAME_FEATURE_DIM
from src.models.model import SequenceRNNClassifier

with open("configs/inference.yaml", encoding="utf-8") as f:
    INFER_CFG = yaml.safe_load(f)["infer"]

def load_metadata(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

class FeatureBuilder:
    """Chuyển đổi danh sách landmarks của MediaPipe thành vector đặc trưng (Left + Right)."""
    def __init__(self, max_hands: int = 2):
        self.max_hands = max_hands

    def build_frame_features(self, hands: List[Dict]) -> List[float]:
        left = [0.0] * HAND_FEATURE_DIM
        right = [0.0] * HAND_FEATURE_DIM
        
        unknown_slot = 0
        for hand in hands:
            flat = normalize_landmarks(hand)
            if flat is None:
                continue
                
            handedness = get_handedness(hand)
            if handedness == "left":
                left = flat
            elif handedness == "right":
                right = flat
            else:
                # Nếu không xác định được bên, ưu tiên điền vào slot trống
                if unknown_slot == 0:
                    left = flat
                else:
                    right = flat
                unknown_slot += 1
                
        return left + right

def sample_to_sequence(frames_buffer: List[List[float]], seq_len: int) -> Tuple[np.ndarray, torch.Tensor]:
    """Chuyển buffer các frame thành mảng numpy và tạo mask độ dài."""
    arr = np.array(frames_buffer, dtype=np.float32)
    current_len = len(arr)
    
    # Tạo mảng output cố định với seq_len
    out = np.zeros((seq_len, FRAME_FEATURE_DIM), dtype=np.float32)
    n = min(current_len, seq_len)
    out[:n, :] = arr[:n, :]
    
    # Mask độ dài thực tế để RNN xử lý chính xác
    length_tensor = torch.LongTensor([n])
    return out, length_tensor

def build_inference_objects(
    model_path: Path | None = None,
    meta_path: Path | None = None,
) -> Tuple[SequenceRNNClassifier, FeatureBuilder, int, str, Dict[int, str]]:

    model_path = model_path or Path(INFER_CFG["model_path"])
    meta_path = meta_path or Path(INFER_CFG["meta_path"])

    model = SequenceRNNClassifier.load(model_path)
    model.eval() # Chuyển sang chế độ inference
    
    meta = load_metadata(meta_path)
    
    # Đọc thông số từ metadata của dataset
    label_map = json.loads(meta["label_map"]) if isinstance(meta["label_map"], str) else meta["label_map"]
    id_to_label = {int(v): str(k) for k, v in label_map.items()}
    
    seq_len = int(meta.get("max_len", 30)) # Mặc định 30 nếu không có
    pad_mode = "zero"
    
    feature_builder = FeatureBuilder()

    return model, feature_builder, seq_len, pad_mode, id_to_label

def smooth_label(history: Deque[int], id_to_label: Dict[int, str]) -> str:
    if not history:
        return ""
    counts: Dict[int, int] = {}
    for idx in history:
        counts[idx] = counts.get(idx, 0) + 1
    best_id = max(counts.items(), key=lambda x: x[1])[0]
    return id_to_label.get(best_id, "")

class StreamingPredictor:
    def __init__(
        self,
        model: SequenceRNNClassifier,
        feature_builder: FeatureBuilder,
        seq_len: int,
        pad_mode: str,
        id_to_label: Dict[int, str],
        record_fps: float | None = None,
        min_history: float | None = None,
        smooth: int | None = None,
        silent_when_no_hands: bool | None = None,
    ) -> None:
        self.model = model
        self.feature_builder = feature_builder
        self.seq_len = seq_len
        self.id_to_label = id_to_label
        self.silent_when_no_hands = silent_when_no_hands

        # Lưu trữ các vector đặc trưng thay vì raw dict để tăng tốc
        self.frames_buffer: Deque[List[float]] = deque(maxlen=seq_len)
        self.pred_history: Deque[int] = deque(maxlen=max(1, int(smooth or 5)))
        self.min_frames_for_pred = max(5, int((min_history or 0.5) * (record_fps or 30)))

    def reset(self) -> None:
        self.frames_buffer.clear()
        self.pred_history.clear()

    def update(self, hands: List[Dict]) -> str:
        # 1. Trích xuất feature từ frame hiện tại
        feat = self.feature_builder.build_frame_features(hands)
        self.frames_buffer.append(feat)

        # 2. Kiểm tra điều kiện dự đoán
        if len(self.frames_buffer) < self.min_frames_for_pred:
            return ""

        # 3. Chuẩn bị dữ liệu cho model (Batch size = 1)
        seq_arr, length_tensor = sample_to_sequence(list(self.frames_buffer), self.seq_len)
        X = torch.from_numpy(seq_arr).unsqueeze(0) # [1, seq_len, feat_dim]

        # 4. Dự đoán
        with torch.no_grad():
            logits = self.model(X, length_tensor)
            pred_id = int(torch.argmax(logits, dim=1).item())

        # 5. Làm mượt kết quả
        self.pred_history.append(pred_id)
        pred_label = smooth_label(self.pred_history, self.id_to_label)

        if self.silent_when_no_hands and len(hands) == 0:
            return ""

        return pred_label