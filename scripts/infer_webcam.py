from __future__ import annotations

import time
from pathlib import Path

import cv2
import yaml

from src.inference.predict import StreamingPredictor, build_inference_objects
from src.utils.overlay import draw_overlay
from src.utils.tts_worker import TTSWorker
from src.utils.webcam import Webcam


with open("configs/inference.yaml", encoding="utf-8") as f:
    INFER_CFG = yaml.safe_load(f)["infer"]


def compute_hand_motion(prev_hands, curr_hands) -> float:

    if not prev_hands or not curr_hands:
        return 0.0

    try:
        prev_landmarks = prev_hands[0]["landmarks"]
        curr_landmarks = curr_hands[0]["landmarks"]
    except Exception:
        return 0.0

    if len(prev_landmarks) != len(curr_landmarks):
        return 0.0

    total = 0.0
    count = 0

    for p, c in zip(prev_landmarks, curr_landmarks):
        px, py = float(p[0]), float(p[1])
        cx, cy = float(c[0]), float(c[1])

        dx = cx - px
        dy = cy - py

        total += (dx * dx + dy * dy) ** 0.5
        count += 1

    if count == 0:
        return 0.0

    return total / count


def main() -> None:

    model, feature_builder, seq_len, pad_mode, id_to_label = build_inference_objects(
        model_path=Path(INFER_CFG["model_path"]),
        meta_path=Path(INFER_CFG["meta_path"]),
    )

    from src.utils.hand_detector import HandDetector

    cam = Webcam(camera_index=INFER_CFG["camera_index"])
    detector = HandDetector()

    predictor = StreamingPredictor(
        model=model,
        feature_builder=feature_builder,
        seq_len=seq_len,
        pad_mode=pad_mode,
        id_to_label=id_to_label,
        record_fps=INFER_CFG["record_fps"],
        min_history=INFER_CFG["min_history"],
        smooth=INFER_CFG["smooth"],
        silent_when_no_hands=INFER_CFG["silent_when_no_hands"],
    )

    tts_worker = TTSWorker()
    tts_worker.start()

    logical_start = time.time()
    prev_loop_time = time.time()
    display_fps = 0.0

    silence_run = 0
    reset_after_silence_frames = INFER_CFG["reset_after_silence_frames"]

    prev_hands = None
    still_run = 0
    stillness_threshold = INFER_CFG["stillness_threshold"]
    reset_after_still_frames = INFER_CFG["reset_after_still_frames"]

    pred_label = ""

    print("Live ASL inference started")
    print("Press ESC to quit")

    try:
        while True:

            frame = cam.read()
            if frame is None:
                print("Cannot receive frame from webcam")
                break

            now = time.time()
            loop_dt = now - prev_loop_time

            if loop_dt > 0:
                display_fps = 1.0 / loop_dt

            prev_loop_time = now
            timestamp_ms = int((now - logical_start) * 1000)

            result = detector.detect(frame, timestamp_ms=timestamp_ms)
            hands = detector.get_hands_data(result, frame.shape)
            has_hands = len(hands) > 0

            pred_label = predictor.update(hands)

            is_silence_state = (
                (not has_hands)
                or (not pred_label)
                or (pred_label.upper() == "SILENCE")
            )

            if is_silence_state:
                silence_run += 1
            else:
                silence_run = 0

            if silence_run == reset_after_silence_frames:
                predictor.reset()
                tts_worker.reset_speech_state()
                pred_label = ""

            motion = compute_hand_motion(prev_hands, hands)
            prev_hands = hands

            if has_hands and motion < stillness_threshold:
                still_run += 1
            else:
                still_run = 0

            if still_run == reset_after_still_frames:
                predictor.reset()
                tts_worker.reset_speech_state()
                pred_label = ""

            if pred_label and pred_label.upper() != "SILENCE":
                tts_worker.request_speak(pred_label)

            frame_with_landmarks = detector.draw_hands(
                frame.copy(),
                detector.last_result,
            )

            vis_frame = draw_overlay(
                frame=frame_with_landmarks,
                hands_count=len(hands),
                fps=display_fps,
                pred_label=pred_label,
            )

            cv2.imshow("ASL Live Inference", vis_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break

    finally:

        tts_worker.stop()
        detector.close()
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()