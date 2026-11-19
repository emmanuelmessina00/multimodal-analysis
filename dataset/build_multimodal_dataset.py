import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd

from utils.video_utils import build_aligned_windows, natural_sort_key
from models.visual.visual_extractor import VisualEmotionExtractor

import cv2
import wandb
from models.visual.visual_extractor import EMOTION_CLASSES

#Nuova aggiunta prendendo spunto dal paper di Shu
def add_neutral_baseline(df: pd.DataFrame, neutral_thresh: float = 0.70) -> pd.DataFrame:
    df = df.copy()
    is_neutral = df["p_neutral"] > neutral_thresh

    for col in ["mu_hr", "rmssd", "pnn50", "sd1"]:

        baseline = df.loc[is_neutral, col].median() if is_neutral.sum() >= 3 else df[col].median()
        df[f"delta_{col}"] = df[col] - baseline

    return df

def build_multimodal_dataset(
        video_paths: list[str],
        stop_video_index: int,
        stop_time_in_segment: float,
        cardiac_path: str,
        window_sec: int = 20,
        weights_path: str = os.path.join(PROJECT_ROOT, "checkpoints", "best.pt"),
) -> pd.DataFrame:
    """
    1. Allinea video e battito e genera le finestre da `window_sec` (già filtrate:
       quelle senza dati cardiaci validi non arrivano qui).
    2. Estrae le probabilità di emozione solo su quelle finestre.
    3. Unisce i due rami sul `window_id`.
    """
    windows = build_aligned_windows(
        video_paths=video_paths,
        stop_video_index=stop_video_index,
        stop_time_in_segment=stop_time_in_segment,
        cardiac_path=cardiac_path,
        window_sec=window_sec,
    )
    if windows.empty:
        raise ValueError("Nessuna finestra valida dopo l'allineamento: niente da estrarre dal video.")

    extractor = VisualEmotionExtractor(weights_path=weights_path)
    df_visual = extractor.extract_windows(video_paths, windows)

    df_merged= (windows
            .merge(df_visual, on='window_id', how='inner', validate='one_to_one')
            .sort_values('ts_start')
            .reset_index(drop=True))

    return add_neutral_baseline(df_merged)


if __name__ == "__main__":
    import glob
    import re

    video_dir = os.path.join(PROJECT_ROOT, "data", "video")
    test_videos = sorted(
        [p for p in glob.glob(os.path.join(video_dir, "*")) if p.lower().endswith(".mp4")],
        key=natural_sort_key
    )
    if not test_videos:
        raise FileNotFoundError(f"Nessun file .mp4 trovato in: {video_dir}")

    print(f"Trovati {len(test_videos)} segmenti video: {[os.path.basename(v) for v in test_videos]}")

    test_cardiac_dir = os.path.join(PROJECT_ROOT, "data", "cardiac", "processed")
    out = os.path.join("data","processed","aligned_data.csv")
    df_multimodal = build_multimodal_dataset(
        video_paths=test_videos,
        stop_video_index=len(test_videos) - 1,
        stop_time_in_segment=725,
        cardiac_path=test_cardiac_dir,
    )
    df_multimodal.to_csv(out, index=False)
    print(f"Salvate {len(df_multimodal)} finestre in {out}")

    wandb.init(project="multimodal-analysis", name="home-session")

    class_labels = dict(enumerate(EMOTION_CLASSES))
    label_to_id = {name: i for i, name in class_labels.items()}

    table = wandb.Table(columns=[
        "window_id", "frame", "label", "conf",
        "mu_hr", "rmssd", "pnn50", "hr_range", "sdnn", "sd1", "sd2", "sd1_sd2",
        "hr_coverage",
        "p_neutral", "max_p_happy", "max_p_surprise", "max_p_sad",
        "ear_mean", "mar_max", "brow_raise_mean", "head_motion_std",
        "bs_smile_mean", "bs_jaw_open_max",
    ])

    caps = [cv2.VideoCapture(v) for v in test_videos]

    for _, row in df_multimodal.iterrows():
        img = None
        v_idx, f_idx = int(row["keyframe_video_idx"]), int(row["keyframe_local_idx"])
        if v_idx >= 0 and f_idx >= 0:
            caps[v_idx].set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ok, frame = caps[v_idx].read()
            if ok:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                boxes = {
                    "predictions": {
                        "box_data": [{
                            "position": {
                                "minX": float(row["bbox_x1"]),
                                "minY": float(row["bbox_y1"]),
                                "maxX": float(row["bbox_x2"]),
                                "maxY": float(row["bbox_y2"]),
                            },
                            "domain": "pixel",
                            "class_id": label_to_id.get(row["keyframe_label"], 4),
                            "box_caption": f"{row['keyframe_label']} ({row['keyframe_conf']:.2f})",
                        }],
                        "class_labels": class_labels,
                    }
                }
                img = wandb.Image(frame_rgb, boxes=boxes)

        table.add_data(
            int(row["window_id"]), img, row["keyframe_label"], float(row["keyframe_conf"]),
            float(row["mu_hr"]), float(row["rmssd"]), float(row["pnn50"]),
            float(row["hr_range"]), float(row["sdnn"]), float(row["sd1"]), float(row["sd2"]), float(row["sd1_sd2"]),
            float(row["hr_coverage"]),
            float(row["p_neutral"]), float(row["max_p_happy"]), float(row["max_p_surprise"]), float(row["max_p_sad"]),
            float(row["ear_mean"]), float(row["mar_max"]), float(row["brow_raise_mean"]), float(row["head_motion_std"]),
            float(row["bs_smile_mean"]), float(row["bs_jaw_open_max"]),
        )

    for cap in caps:
        cap.release()

    wandb.log({"multimodal_table": table})
    wandb.finish()