from __future__ import annotations

import queue
import threading
import yaml


with open("configs/utils.yaml", encoding="utf-8") as f:
    utils_cfg = yaml.safe_load(f)

with open("configs/data.yaml", encoding="utf-8") as f:
    data_cfg = yaml.safe_load(f)

TTS_CFG = utils_cfg["tts"]
DATA_CFG = data_cfg["label"]


class TTSWorker:
    def __init__(self, max_queue_size: int | None = None) -> None:

        max_queue_size = max_queue_size or TTS_CFG["max_queue_size"]

        self.queue: "queue.Queue[str]" = queue.Queue(maxsize=max_queue_size)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

        self.last_spoken_label: str = ""
        self.last_requested_label: str = ""

    def start(self) -> None:
        self.thread.start()

    def request_speak(self, label: str) -> None:

        if not label:
            return

        if label.upper() == DATA_CFG["silence_label"]:
            return

        if label == self.last_requested_label:
            return

        self.last_requested_label = label

        try:
            self.queue.put_nowait(label)
        except queue.Full:
            try:
                _ = self.queue.get_nowait()
            except queue.Empty:
                pass

            try:
                self.queue.put_nowait(label)
            except queue.Full:
                pass

    def reset_speech_state(self) -> None:
        self.last_requested_label = ""
        self.last_spoken_label = ""

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=1.0)

    def _speak_blocking(self, text: str) -> None:

        try:
            import pyttsx3
        except Exception:
            return

        try:
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
            engine.stop()
        except Exception:
            pass

    def _run(self) -> None:

        while not self.stop_event.is_set():

            try:
                label = self.queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if not label:
                continue

            if label.upper() == DATA_CFG["silence_label"]:
                continue

            if label == self.last_spoken_label:
                continue

            self._speak_blocking(label)
            self.last_spoken_label = label