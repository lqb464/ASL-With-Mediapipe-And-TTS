import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import yaml


with open("configs/utils.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

HD_CFG = cfg["hand_detector"]


class HandDetector:
    def __init__(
        self,
        model_path: str | None = None,
        num_hands: int | None = None,
        min_hand_detection_confidence: float | None = None,
        min_hand_presence_confidence: float | None = None,
        min_tracking_confidence: float | None = None,
    ):

        model_path = model_path or HD_CFG["model_path"]
        num_hands = num_hands or HD_CFG["num_hands"]
        min_hand_detection_confidence = (
            min_hand_detection_confidence or HD_CFG["min_hand_detection_confidence"]
        )
        min_hand_presence_confidence = (
            min_hand_presence_confidence or HD_CFG["min_hand_presence_confidence"]
        )
        min_tracking_confidence = (
            min_tracking_confidence or HD_CFG["min_tracking_confidence"]
        )

        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(f"Missing model: {model_file}")

        base_options = python.BaseOptions(model_asset_path=str(model_file))
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=min_hand_detection_confidence,
            min_hand_presence_confidence=min_hand_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self.last_result = None

    def detect(self, frame, timestamp_ms=None):
        if timestamp_ms is None:
            timestamp_ms = int(time.time() * 1000)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        self.last_result = result
        return result

    def get_hands_data(self, result, frame_shape):
        hands_data = []
        if result is None or not result.hand_landmarks:
            return hands_data

        h, w, _ = frame_shape

        for i, hand_landmarks in enumerate(result.hand_landmarks):
            points = [[int(lm.x * w), int(lm.y * h)] for lm in hand_landmarks]

            label = None
            score = None

            if result.handedness and i < len(result.handedness):
                label = result.handedness[i][0].category_name
                score = result.handedness[i][0].score

            hands_data.append(
                {
                    "label": label,
                    "score": score,
                    "landmarks": points,
                }
            )

        label_order = {"Left": 0, "Right": 1, None: 2}
        hands_data.sort(key=lambda hand: label_order.get(hand["label"], 2))

        return hands_data

    def draw_hands(self, frame, result):
        """
        Vẽ landmarks và nối các điểm bằng OpenCV thuần.
        """
        if result is None or not result.hand_landmarks:
            return frame

        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (5, 9), (9, 10), (10, 11), (11, 12),
            (9, 13), (13, 14), (14, 15), (15, 16),
            (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
        ]

        hands_data = self.get_hands_data(result, frame.shape)

        for hand_data in hands_data:
            points = hand_data["landmarks"]
            label = hand_data["label"]
            score = hand_data["score"]

            points_tuple = [(p[0], p[1]) for p in points]

            for start_idx, end_idx in connections:
                cv2.line(frame, points_tuple[start_idx], points_tuple[end_idx], (0, 255, 0), 2)

            for x, y in points_tuple:
                cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)

            text = label or "Unknown"
            if score is not None:
                text += f" {score:.2f}"

            cv2.putText(
                frame,
                text,
                (points_tuple[0][0] + 10, points_tuple[0][1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2,
            )

        return frame

    def close(self):
        self.landmarker.close()