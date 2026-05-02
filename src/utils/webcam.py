import cv2
import yaml


with open("configs/utils.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

CAM_CFG = cfg["webcam"]


class Webcam:
    def __init__(
        self,
        camera_index=None,
        width=None,
        height=None,
        fps=None,
    ):

        camera_index = camera_index if camera_index is not None else CAM_CFG["camera_index"]
        width = width if width is not None else CAM_CFG["width"]
        height = height if height is not None else CAM_CFG["height"]
        fps = fps if fps is not None else CAM_CFG["fps"]

        self.cap = cv2.VideoCapture(camera_index)

        if not self.cap.isOpened():
            raise RuntimeError("Cannot open webcam")

        if width is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)

        if height is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        if fps is not None:
            self.cap.set(cv2.CAP_PROP_FPS, fps)

    def read(self):

        ret, frame = self.cap.read()

        if not ret:
            return None

        frame = cv2.flip(frame, 1)

        return frame

    def get_actual_fps(self):

        fps = self.cap.get(cv2.CAP_PROP_FPS)

        if fps is None or fps <= 0:
            return None

        return fps

    def release(self):
        self.cap.release()