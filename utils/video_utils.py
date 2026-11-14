import cv2
import pandas as pd
from models.cardiac.cardiac_extractor import extract_cardiac_window

def build_backward_aligned_windows(
        video_paths: list[str],
        stop_video_index: int,  # 0 se lo stop è nel primo video, 1 se è nel secondo
        stop_time_in_segment: float,  # secondi dall'inizio di quel segmento
        cardiac_csv_path: str,
        window_sec: int = 20
):

    durations = []
    fps_list = []

    #Prendiamo le info utili dei video
    for vp in video_paths:
        cap = cv2.VideoCapture(vp)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        durations.append(frames / fps)
        fps_list.append(fps)
        cap.release()

    t_stop_vid = sum(durations[:stop_video_index]) + stop_time_in_segment
    print(f"Istante di stop sul video continuo: {t_stop_vid:.2f} s")


    df_hr = pd.read_csv(cardiac_csv_path).dropna(subset=['hr']).reset_index(drop=True)
    total_hr_sec = len(df_hr)

    if total_hr_sec < t_stop_vid:
        raise ValueError(f"Il battito ({total_hr_sec}s) è più corto del video fino allo stop ({t_stop_vid:.1f}s)!")

    #Prendiamo solo gli ultimi secondi che ci interessano in modo da allinearci con lo spegnimento del dispositivo
    start_hr_idx = int(round(total_hr_sec - t_stop_vid))
    df_hr_aligned = df_hr.iloc[start_hr_idx:].reset_index(drop=True)

    # Calcolo il numero di finestre possibili, e un punto di inizio valido in modo non avere ad esempio una finestra residua da 5s e le altre da 20s
    n_windows = int(t_stop_vid // window_sec)
    valid_start_vid = t_stop_vid - (n_windows * window_sec)

    windows = []
    for w in range(n_windows):
        v_start = valid_start_vid + w * window_sec
        v_end = v_start + window_sec

        hr_s = int(round(v_start))
        mu_hr, rmssd, pnn50 = extract_cardiac_window(
            df_hr_aligned, start_sec=hr_s, duration_sec=window_sec
        )

        windows.append({
            "window_id": w,
            "vid_start_sec": v_start,
            "vid_end_sec": v_end,
            "mu_hr": mu_hr,
            "rmssd": rmssd,
            "pnn50": pnn50,
        })

    return pd.DataFrame(windows), durations, fps_list