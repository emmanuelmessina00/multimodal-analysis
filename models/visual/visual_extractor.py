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


def extract_frames(video_path: str, start_sec: float = 0.0, duration_sec: float = 20.0):

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"[ERRORE] Video non trovato in: {video_path}")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    start_frame = int(start_sec * fps)
    total_frames = int(duration_sec * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames = []
    for _ in range(total_frames):
        ret, frm = cap.read()
        if not ret:
            break
        frames.append(frm)
    cap.release()
    return frames, fps


class VisualEmotionExtractor:
    """
    Wrapper pulito attorno al modello YOLO migliore di Sotelo (YOLO26s_640).
    Prende una finestra video e restituisce il vettore delle probabilità visive (6 valori)
    pronto per la Late Fusion col battito cardiaco.
    """
    def __init__(self, weights_path: str = "checkpoints/best.pt", conf: float = 0.25, imgsz: int = 640, use_clahe: bool = False):
        self.weights_path = weights_path
        self.conf = conf
        self.imgsz = imgsz
        self.use_clahe = use_clahe
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        if os.path.exists(weights_path):
            print(f"[VisualBranch] Caricamento pesi YOLO da: {weights_path} su {self.device}")
            self.model = YOLO(weights_path)
        else:
            print(f"[WARN] Pesi '{weights_path}' non trovati! Uso modalità simulata (dummy) per testare la pipeline.")
            self.model = None

    def extract_window_probabilities(self, video_path: str = None, start_sec: float = 0.0, duration_sec: float = 20.0) -> np.ndarray:
        """
        Analizza la finestra video [start_sec, start_sec + duration_sec] e restituisce
        un array NumPy di 6 elementi con la distribuzione di probabilità delle emozioni:
        [p_angry, p_happy, p_sad, p_surprise, p_neutral, p_fear]
        """

        if self.model is None or video_path is None or not os.path.exists(video_path):
            # Ordine: [angry, happy, sad, surprise, neutral, fear]
            dummy_probs = np.array([0.01, 0.05, 0.01, 0.01, 0.90, 0.02], dtype=np.float32)
            return dummy_probs

        frames, _ = extract_frames(video_path, start_sec, duration_sec)
        if not frames:
            return np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)

        # Accumulatore delle confidenze per le 6 classi sulla finestra
        conf_accumulator = np.zeros(len(EMOTION_CLASSES), dtype=np.float32)
        detected_frames = 0

        for frm in frames:
            if self.use_clahe:
                frm = apply_clahe(frm)

            result = self.model.predict(
                source=frm,
                conf=self.conf,
                imgsz=self.imgsz,
                device=self.device,
                verbose=False,
                save=False
            )[0]

            if len(result.boxes) > 0:
                detected_frames += 1
                # Prende il volto principale rilevato nel frame
                box = result.boxes[0]
                cls_id = int(box.cls[0])
                conf_val = float(box.conf[0])
                if 0 <= cls_id < len(EMOTION_CLASSES):
                    conf_accumulator[cls_id] += conf_val

        # Se non ha rilevato nessun volto (es. testa abbassata sul tavolo), restituisce tutto su 'neutral'
        if detected_frames == 0 or conf_accumulator.sum() == 0:
            return np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)

        # Normalizza affinché la somma delle probabilità faccia 1.0
        probs = conf_accumulator / conf_accumulator.sum()
        return probs

    def get_all_video_windows(
        self,
        video_path: str,
        duration_sec: int = 20,
        stride_sec: int = 20,
    ) :

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Impossibile aprire il video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()

        video_duration = total_frames / fps
        rows = []


        for start_t in range(0, int(video_duration) - duration_sec + 1, stride_sec):
            end_t = start_t + duration_sec


            probs = self.extract_window_probabilities(
                video_path=video_path, start_sec=start_t, duration_sec=duration_sec
            )

            row_dict = {
                "window_start": start_t,
                "window_end": end_t,
            }
            for emo, p in zip(EMOTION_CLASSES, probs):
                row_dict[f"p_{emo}"] = float(p)

            rows.append(row_dict)

        return pd.DataFrame(rows)

if __name__ == "__main__":
    extractor = VisualEmotionExtractor(weights_path="checkpoints/best.pt")
    test_video = "data/video/camera_D_0_08.mp4"


    df_video = extractor.get_all_video_windows(
        video_path=test_video, duration_sec=20, stride_sec=20
    )

    print(
        f"\n--- Estratte {len(df_video)} finestre video da {test_video} ---"
    )
    print(df_video)