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
            print(f"[WARN] Pesi '{weights_path}' non trovati! Uso modalita' simulata (dummy).")
            self.model = None

    def _predict(self, frames: list[np.ndarray]) -> list[tuple[int, float] | None]:
        """Per ogni frame ritorna (classe, confidenza) del primo volto rilevato, o None."""
        if not frames or self.model is None:
            return [None] * len(frames)
        if self.use_clahe:
            frames = [apply_clahe(f) for f in frames]

        results = self.model.predict(source=frames, conf=self.conf, imgsz=self.imgsz,
                                     device=self.device, verbose=False, save=False, batch=len(frames))
        out = []
        for res in results:
            if len(res.boxes) > 0 and 0 <= int(res.boxes[0].cls[0]) < len(EMOTION_CLASSES):
                out.append((int(res.boxes[0].cls[0]), float(res.boxes[0].conf[0])))
            else:
                out.append(None)
        return out

    def extract_windows(
        self,
        video_paths: list[str],
        windows: pd.DataFrame,
        target_fps: int = 4,
        batch_size: int = 16,
    ) -> pd.DataFrame:
        """
        Calcola le probabilità di emozione per ogni finestra della tabella `windows`
        (colonne richieste: window_id, vid_start_sec, vid_end_sec sulla timeline continua
        dei segmenti video, nello stesso ordine di `video_paths`).

        Un'unica passata sequenziale su tutti i segmenti: le finestre a cavallo tra due
        file ricevono frame da entrambi. I frame fuori dalle finestre non vengono convertiti né analizzati.
        """
        win = windows.sort_values('vid_start_sec')
        starts = win['vid_start_sec'].to_numpy()
        ends = win['vid_end_sec'].to_numpy()
        ids = win['window_id'].to_numpy()

        conf_acc = {wid: np.zeros(len(EMOTION_CLASSES), dtype=np.float32) for wid in ids}
        buffer_frames, buffer_ids = [], []

        def flush():
            for wid, pred in zip(buffer_ids, self._predict(buffer_frames)):
                if pred is not None:
                    conf_acc[wid][pred[0]] += pred[1]
            buffer_frames.clear()
            buffer_ids.clear()

        offset = 0.0
        for vp in video_paths:
            cap = cv2.VideoCapture(vp)
            if not cap.isOpened():
                raise FileNotFoundError(f"Impossibile aprire il video: {vp}")
            fps = cap.get(cv2.CAP_PROP_FPS)
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            step = max(1, int(round(fps / target_fps)))
            print(f"[VisualBranch] {os.path.basename(vp)}: {n_frames / fps:.1f}s @ {fps:.2f} fps, "
                  f"1 frame ogni {step}")

            for frame_idx in range(n_frames):
                if not cap.grab():  # avanza; la conversione del frame avviene solo con retrieve()
                    break
                if frame_idx % step:
                    continue
                t = offset + frame_idx / fps
                k = np.searchsorted(starts, t, side='right') - 1
                if k < 0 or t >= ends[k]:
                    continue  # frame fuori da ogni finestra valida
                ok, frame = cap.retrieve()
                if not ok:
                    continue
                buffer_frames.append(frame)
                buffer_ids.append(ids[k])
                if len(buffer_frames) >= batch_size:
                    flush()

            cap.release()
            offset += n_frames / fps
        flush()

        rows = []
        for wid in ids:
            acc = conf_acc[wid]
            # Fallback: nessun volto rilevato nella finestra -> neutral
            probs = acc / acc.sum() if acc.sum() > 0 else np.eye(len(EMOTION_CLASSES))[EMOTION_CLASSES.index("neutral")]
            rows.append({'window_id': wid, **{f"p_{e}": float(p) for e, p in zip(EMOTION_CLASSES, probs)}})
        return pd.DataFrame(rows)


def fixed_windows(video_path: str, window_sec: int = 20) -> pd.DataFrame:
    """Finestre consecutive da 0 s su un singolo video (utile per test senza dati cardiaci)."""
    cap = cv2.VideoCapture(video_path)
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    n = int(duration // window_sec)
    return pd.DataFrame({'window_id': range(n),
                         'vid_start_sec': [i * window_sec for i in range(n)],
                         'vid_end_sec': [(i + 1) * window_sec for i in range(n)]})


if __name__ == "__main__":
    extractor = VisualEmotionExtractor(weights_path="checkpoints/best.pt")
    test_video = "data/video/camera_D_0_08.mp4"

    df_video = extractor.extract_windows([test_video], fixed_windows(test_video), target_fps=4)

    print(f"\n--- Estratte {len(df_video)} finestre video da {test_video} ---")
    print(df_video)
