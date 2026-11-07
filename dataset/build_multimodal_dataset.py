import pandas as pd

from models.cardiac.cardiac_extractor import get_all_cardiac_windows
from models.visual.visual_extractor import VisualEmotionExtractor


def get_multimodal_dataset(video_path, cardiac_path):
    cardiac_df = pd.read_csv('data/cardiac/example.csv')
    cardiac_w_df=get_all_cardiac_windows(cardiac_df)

    extractor = VisualEmotionExtractor(weights_path="checkpoints/best.pt")
    test_video = "data/video/camera_D_0_08.mp4"


    video_w_df = extractor.get_all_video_windows(
        video_path=test_video, duration_sec=20, stride_sec=20
    )

    df_multimodal = pd.merge(
        video_w_df,
        cardiac_w_df,
        on=["window_start", "window_end"],
        how="inner"
    )

    return df_multimodal
if __name__ == "__main__":
    df_multimodal=get_multimodal_dataset(video_path="data/video/camera_D_0_08.mp4",cardiac_path="data/cardiac/example.csv")
    df_multimodal.to_csv("data/multimodal.csv", index=False)