import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd

from utils.video_utils import build_aligned_windows
from models.visual.visual_extractor import VisualEmotionExtractor


def build_multimodal_dataset(
        video_paths: list[str],
        stop_video_index: int,
        stop_time_in_segment: float,
        cardiac_path: str,
        window_sec: int = 20,
        weights_path: str = os.path.join(PROJECT_ROOT, "checkpoints", "best.pt"),
        hr_stop_ts=None,
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
        hr_stop_ts=hr_stop_ts,
    )
    if windows.empty:
        raise ValueError("Nessuna finestra valida dopo l'allineamento: niente da estrarre dal video.")

    extractor = VisualEmotionExtractor(weights_path=weights_path)
    df_visual = extractor.extract_windows(video_paths, windows)

    return (windows
            .merge(df_visual, on='window_id', how='inner', validate='one_to_one')
            .sort_values('ts_start')
            .reset_index(drop=True))


if __name__ == "__main__":
    VIDEO_PATHS = [
        os.path.join(PROJECT_ROOT, "data", "video", "camera_A000.mp4"),
        os.path.join(PROJECT_ROOT, "data", "video", "camera_A001.mp4"),
    ]
    STOP_VIDEO_INDEX = 1
    STOP_TIME_IN_SEGMENT = 120.0
    CARDIAC_PATH = os.path.join(PROJECT_ROOT, "data", "cardiac")
    OUT_CSV = os.path.join(PROJECT_ROOT, "data", "multimodal.csv")

    df_multimodal = build_multimodal_dataset(
        video_paths=VIDEO_PATHS,
        stop_video_index=STOP_VIDEO_INDEX,
        stop_time_in_segment=STOP_TIME_IN_SEGMENT,
        cardiac_path=CARDIAC_PATH,
    )
    df_multimodal.to_csv(OUT_CSV, index=False)
    print(f"Salvate {len(df_multimodal)} finestre in {OUT_CSV}")