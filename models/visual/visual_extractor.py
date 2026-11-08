import os
import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

EMOTION_CLASSES = ["angry", "happy", "sad", "surprise", "neutral", "fear"]


def apply_clahe(frame: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)
    merged_lab = cv2.merge((cl, a_channel, b_channel))
    return cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)


class VisualEmotionExtractor:
    """
    Wrapper attorno al modello YOLO (Sotelo YOLO26s_640).
    Estrae finestre temporali campionando i frame ed eseguendo inferenza a batch.
    """

    def __init__(self, weights_path: str = "checkpoints/best.pt", conf: float = 0.25, imgsz: int = 640,
                 use_clahe: bool = False):
        self.weights_path = weights_path
        self.conf = conf
        self.imgsz = imgsz
        self.use_clahe = use_clahe
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        if os.path.exists(weights_path):
            print(f"[VisualBranch] Caricamento pesi YOLO da: {weights_path} su {self.device}")
            self.model = YOLO(weights_path)
        else:
            print(f"[WARN] Pesi '{weights_path}' non trovati! Uso modalita' simulata (dummy).")
            self.model = None

    def _process_batch(self, batch_frames: list[np.ndarray]) -> tuple[np.ndarray, int]:
        """Esegue inferenza su un batch di frame accumulando le confidenze."""
        conf_acc = np.zeros(len(EMOTION_CLASSES), dtype=np.float32)
        detected = 0

        if not batch_frames or self.model is None:
            return conf_acc, detected

        if self.use_clahe:
            batch_frames = [apply_clahe(f) for f in batch_frames]

        # Inferenza batch su GPU
        results = self.model.predict(
            source=batch_frames,
            conf=self.conf,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
            save=False,
            batch=len(batch_frames)
        )

        for res in results:
            if len(res.boxes) > 0:
                detected += 1
                box = res.boxes[0]
                cls_id = int(box.cls[0])
                conf_val = float(box.conf[0])
                if 0 <= cls_id < len(EMOTION_CLASSES):
                    conf_acc[cls_id] += conf_val

        return conf_acc, detected

    def get_all_video_windows(
            self,
            video_path: str,
            duration_sec: int = 20,
            stride_sec: int = 20,
            target_fps: int = 4,  # Campiona 4 frame al secondo anziche' 25/30
            batch_size: int = 16  # Batching per sfruttare la GPU T4
    ) -> pd.DataFrame:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"[ERRORE] Video non trovato in: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Impossibile aprire il video: {video_path}")

        orig_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = total_frames / orig_fps

        frame_step = max(1, int(round(orig_fps / target_fps)))
        frames_per_window = int(duration_sec * orig_fps)

        print(
            f"[VisualBranch] FPS nativo: {orig_fps:.1f} | Subsampling step: ogni {frame_step} frame (~{target_fps} fps)")

        rows = []
        current_start_sec = 0

        while current_start_sec + duration_sec <= int(video_duration):
            window_end_sec = current_start_sec + duration_sec

            # Posizionamento sequenziale efficiente
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(current_start_sec * orig_fps))

            sampled_frames = []
            f_count = 0
            while f_count < frames_per_window:
                ret, frame = cap.read()
                if not ret:
                    break
                if f_count % frame_step == 0:
                    sampled_frames.append(frame)
                f_count += 1

            # Inferenza a batch sulla finestra corrente
            conf_accumulator = np.zeros(len(EMOTION_CLASSES), dtype=np.float32)
            detected_total = 0

            for i in range(0, len(sampled_frames), batch_size):
                b_frames = sampled_frames[i: i + batch_size]
                c_acc, d_count = self._process_batch(b_frames)
                conf_accumulator += c_acc
                detected_total += d_count

            # Normalizzazione probabilità
            if detected_total == 0 or conf_accumulator.sum() == 0:
                probs = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
            else:
                probs = conf_accumulator / conf_accumulator.sum()

            row_dict = {
                "window_start": current_start_sec,
                "window_end": window_end_sec,
            }
            for emo, p in zip(EMOTION_CLASSES, probs):
                row_dict[f"p_{emo}"] = float(p)

            rows.append(row_dict)
            current_start_sec += stride_sec

        cap.release()
        return pd.DataFrame(rows)


if __name__ == "__main__":
    extractor = VisualEmotionExtractor(weights_path="checkpoints/best.pt")
    test_video = "data/video/camera_D_0_08.mp4"

    df_video = extractor.get_all_video_windows(
        video_path=test_video,
        duration_sec=20,
        stride_sec=20,
        target_fps=4,
        batch_size=16
    )

    print(f"\n--- Estratte {len(df_video)} finestre video da {test_video} ---")
    print(df_video)