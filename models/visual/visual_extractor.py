import os
import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO
from models.visual.geometry_extractor import FacialGeometryExtractor
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
        self.geometry_extractor = FacialGeometryExtractor()

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
                bbox = res.boxes[0].xyxy[0].cpu().numpy().tolist()  # [x1, y1, x2, y2]
                out.append((int(res.boxes[0].cls[0]), float(res.boxes[0].conf[0]), bbox))
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
        Un'unica passata sequenziale su tutti i segmenti: le finestre a cavallo tra due
        file ricevono frame da entrambi. I frame fuori dalle finestre non vengono convertiti né analizzati.
        """
        win = windows.sort_values('vid_start_sec')
        starts = win['vid_start_sec'].to_numpy()
        ends = win['vid_end_sec'].to_numpy()
        ids = win['window_id'].to_numpy()

        neutral_idx = EMOTION_CLASSES.index("neutral")
        buffer_frames, buffer_ids, buffer_meta = [], [], []
        frame_probs = {wid: [] for wid in ids}
        sampled_counts = {wid: 0 for wid in ids}
        best_keyframe = {wid: None for wid in ids}
        frame_geom = {wid: [] for wid in ids}  # nuova aggiunta per aggiungere la logica del Geometry Extractor

        def flush():
            for wid, (vp_idx, f_idx), frame, pred in zip(
                    buffer_ids, buffer_meta, buffer_frames, self._predict(buffer_frames)
            ):
                if pred is not None:
                    cls_idx, conf, bbox = pred

                    # Aggiorna lo storico probabilità per mean / max / std
                    vec = np.zeros(len(EMOTION_CLASSES), dtype=np.float32)
                    vec[cls_idx] = conf
                    frame_probs[wid].append(vec)
                    geom = self.geometry_extractor.extract_frame(frame)
                    if geom is not None:
                        frame_geom[wid].append(geom)


                    # Aggiorna il miglior frame della finestra (priorità a emozioni non-neutre)
                    score = (cls_idx != neutral_idx, conf)


                    if best_keyframe[wid] is None or score > best_keyframe[wid]["score"]:
                        best_keyframe[wid] = {
                            "score": score,
                            "video_idx": vp_idx,
                            "local_idx": f_idx,
                            "bbox": bbox,
                            "label": EMOTION_CLASSES[cls_idx],
                            "conf": conf,
                        }

            buffer_frames.clear()
            buffer_ids.clear()
            buffer_meta.clear()

        offset = 0.0
        for vp_idx, vp in enumerate(video_paths):
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
                sampled_counts[ids[k]] += 1
                buffer_frames.append(frame)
                buffer_ids.append(ids[k])
                buffer_meta.append((vp_idx, frame_idx))
                if len(buffer_frames) >= batch_size:
                    flush()
            cap.release()
            offset += n_frames / fps
        flush()
        self.geometry_extractor.close()
        rows = []
        neutral_idx = EMOTION_CLASSES.index("neutral")

        for wid in ids:
            det_list = frame_probs[wid]
            n_sampled = sampled_counts[wid]
            det_rate = len(det_list) / n_sampled if n_sampled > 0 else 0.0

            if len(det_list) > 0:
                M = np.vstack(det_list)  # shape: (n_rilevamenti, 6)
                mean_p = M.mean(axis=0)
                max_p = M.max(axis=0)
                std_p = M.std(axis=0)
            else:
                # Fallback: nessun volto rilevato nella finestra -> neutral = 1.0
                mean_p = np.eye(len(EMOTION_CLASSES), dtype=np.float32)[neutral_idx]
                max_p = mean_p.copy()
                std_p = np.zeros(len(EMOTION_CLASSES), dtype=np.float32)

            row = {
                "window_id": wid,
                "face_detection_rate": float(det_rate),
            }
            for i, e in enumerate(EMOTION_CLASSES):
                row[f"p_{e}"] = float(mean_p[i])
                row[f"max_p_{e}"] = float(max_p[i])
                row[f"std_p_{e}"] = float(std_p[i])

            kf = best_keyframe[wid]
            if kf is not None:
                row["keyframe_video_idx"] = int(kf["video_idx"])
                row["keyframe_local_idx"] = int(kf["local_idx"])
                row["bbox_x1"] = float(kf["bbox"][0])
                row["bbox_y1"] = float(kf["bbox"][1])
                row["bbox_x2"] = float(kf["bbox"][2])
                row["bbox_y2"] = float(kf["bbox"][3])
                row["keyframe_label"] = kf["label"]
                row["keyframe_conf"] = float(kf["conf"])
            else:
                row["keyframe_video_idx"] = -1
                row["keyframe_local_idx"] = -1
                row["bbox_x1"] = np.nan
                row["bbox_y1"] = np.nan
                row["bbox_x2"] = np.nan
                row["bbox_y2"] = np.nan
                row["keyframe_label"] = "none"
                row["keyframe_conf"] = 0.0

            geom_list = frame_geom[wid]
            if len(geom_list) > 0:
                ear_vals = np.array([g["ear"] for g in geom_list], dtype=np.float32)
                mar_vals = np.array([g["mar"] for g in geom_list], dtype=np.float32)
                brow_vals = np.array([g["brow_raise"] for g in geom_list], dtype=np.float32)
                pitch_vals = np.array([g["pitch"] for g in geom_list], dtype=np.float32)
                yaw_vals = np.array([g["yaw"] for g in geom_list], dtype=np.float32)
                roll_vals = np.array([g["roll"] for g in geom_list], dtype=np.float32)

                pitch_std = float(np.std(pitch_vals))
                yaw_std = float(np.std(yaw_vals))
                roll_std = float(np.std(roll_vals))

                row["ear_mean"] = float(np.mean(ear_vals))
                row["ear_std"] = float(np.std(ear_vals))
                row["mar_mean"] = float(np.mean(mar_vals))
                row["mar_max"] = float(np.max(mar_vals))
                row["brow_raise_mean"] = float(np.mean(brow_vals))
                row["pitch_std"] = pitch_std
                row["yaw_std"] = yaw_std
                row["roll_std"] = roll_std
                row["head_motion_std"] = float(np.sqrt(pitch_std ** 2 + yaw_std ** 2 + roll_std ** 2))
                row["bs_smile_mean"] = float(np.mean([g["bs_smile"] for g in geom_list]))
                row["bs_jaw_open_max"] = float(np.max([g["bs_jaw_open"] for g in geom_list]))
                row["bs_brow_up_mean"] = float(np.mean([g["bs_brow_up"] for g in geom_list]))
            else:
                for col in [
                    "ear_mean", "ear_std", "mar_mean", "mar_max", "brow_raise_mean",
                    "pitch_std", "yaw_std", "roll_std", "head_motion_std",
                    "bs_smile_mean", "bs_jaw_open_max", "bs_brow_up_mean",
                ]:
                    row[col] = np.nan

            rows.append(row)

        return pd.DataFrame(rows)

if __name__ == "__main__":
    pass
