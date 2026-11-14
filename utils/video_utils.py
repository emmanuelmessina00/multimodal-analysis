"""
Allineamento video <-> battito a finestre fisse (default 20 s).

Logica:
  1. I segmenti video vengono messi in fila su una timeline continua.
  2. L'istante di stop (indicato a mano nel video) coincide con la fine della registrazione cardiaca.
  3. Da lì si tagliano finestre da `window_sec` a ritroso, fino all'inizio del video.
  4. Le finestre senza dati cardiaci sufficienti (es. buco dovuto a un .fit illeggibile)
     vengono SCARTATE, così non entrano né nel calcolo video né nel dataset finale.

Ogni finestra ha un `window_id` stabile (0 = prima finestra del video): se una finestra viene
scartata, il suo id semplicemente manca, e si vede dove sono i buchi.
"""
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import cv2
import pandas as pd

from models.cardiac.cardiac_extractor import load_cardiac_folder, cardiac_features


def get_video_info(video_paths: list[str]) -> pd.DataFrame:
    """fps, numero di frame, durata e offset di ogni segmento sulla timeline continua."""
    rows, offset = [], 0.0
    for vp in video_paths:
        if not os.path.exists(vp):
            raise FileNotFoundError(f"File video non trovato: {vp}")
        cap = cv2.VideoCapture(vp)
        fps = cap.get(cv2.CAP_PROP_FPS)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if not fps or fps <= 0 or n_frames <= 0:
            # Un fps inventato sposterebbe tutte le finestre: meglio fermarsi
            raise ValueError(f"FPS/numero frame non leggibili per {vp} (fps={fps}, frames={n_frames})")
        duration = n_frames / fps
        rows.append({'path': vp, 'fps': fps, 'n_frames': n_frames,
                     'duration': duration, 'offset': offset})
        offset += duration
    return pd.DataFrame(rows)


def check_cardiac_anchor(files: pd.DataFrame) -> None:
    """
    L'allineamento usa l'ULTIMO campione cardiaco come istante di stop.
    Se l'ultimo file della registrazione è illeggibile/troncato (o non si sa dove si colloca),
    quell'istante non è affidabile e tutte le finestre risulterebbero sfasate: ci fermiamo.
    """
    ok = files[files['status'] == 'ok']
    bad = files[files['status'] != 'ok']
    if ok.empty:
        raise ValueError("Nessun file cardiaco integro: impossibile ancorare lo stop.")

    last_ok_start = pd.to_datetime(ok['start'], utc=True).max()
    bad_starts = pd.to_datetime(bad['start'], utc=True)
    # File danneggiato con inizio sconosciuto o successivo all'ultimo integro -> è (forse) l'ultimo
    suspicious = bad[bad_starts.isna() | (bad_starts >= last_ok_start)]
    if not suspicious.empty:
        raise ValueError(
            "La fine della registrazione cardiaca non è affidabile: l'ultimo file è danneggiato "
            f"o non databile ({', '.join(suspicious['file'])}). "
            "Passa `hr_stop_ts` con l'orario reale di stop dell'orologio, oppure escludi la sessione."
        )
    if not bad.empty:
        print(f"[WARN] File danneggiati in mezzo alla registrazione ({', '.join(bad['file'])}): "
              "le finestre che cadono nel buco verranno scartate.")


def build_aligned_windows(
        video_paths: list[str],
        stop_video_index: int,          # 0 se lo stop è nel primo video, 1 se è nel secondo, ...
        stop_time_in_segment: float,    # secondi dall'inizio di quel segmento
        cardiac_path: str,              # cartella con i file della registrazione o singolo file
        window_sec: int = 20,
        min_hr_coverage: float = 0.8,   # frazione minima della finestra coperta da campioni cardiaci
        hr_stop_ts: str | pd.Timestamp | None = None,  # override manuale dell'istante di stop
) -> pd.DataFrame:
    """
    Ritorna la tabella delle finestre allineate, ordinata nel tempo, con colonne:
    window_id, video_idx, vid_start_sec, vid_end_sec, ts_start, ts_end, mu_hr, rmssd, pnn50, hr_coverage
    (vid_*_sec sono sulla timeline continua di tutti i segmenti video).
    """
    videos = get_video_info(video_paths)
    if stop_time_in_segment > videos['duration'][stop_video_index]:
        raise ValueError("stop_time_in_segment oltre la durata del segmento indicato.")
    t_stop_vid = videos['offset'][stop_video_index] + stop_time_in_segment

    df_hr, files = load_cardiac_folder(cardiac_path)
    if hr_stop_ts is None:
        check_cardiac_anchor(files)
        hr_stop_ts = df_hr['timestamp'].iloc[-1]
    else:
        hr_stop_ts = pd.Timestamp(hr_stop_ts)
        if hr_stop_ts.tzinfo is None:
            hr_stop_ts = hr_stop_ts.tz_localize(df_hr['timestamp'].dt.tz)

    # Istante (orologio) che corrisponde a t=0 della timeline video continua
    video_zero_ts = hr_stop_ts - pd.Timedelta(seconds=t_stop_vid)

    # Finestre a ritroso dallo stop: l'ultima finisce esattamente allo stop
    n_windows = int(t_stop_vid // window_sec)
    first_start = t_stop_vid - n_windows * window_sec
    starts = [round(first_start + i * window_sec, 3) for i in range(n_windows)]

    windows = pd.DataFrame({
        'window_id': range(n_windows),
        'vid_start_sec': starts,
        'vid_end_sec': [s + window_sec for s in starts],
    })
    windows['video_idx'] = (videos['offset'].searchsorted(windows['vid_start_sec'], side='right') - 1)
    windows['ts_start'] = video_zero_ts + pd.to_timedelta(windows['vid_start_sec'], unit='s')
    windows['ts_end'] = video_zero_ts + pd.to_timedelta(windows['vid_end_sec'], unit='s')

    windows = windows.merge(cardiac_features(df_hr, windows, window_sec), on='window_id')

    # Scarta le finestre senza copertura cardiaca sufficiente (buchi, video iniziato prima dell'orologio, ...)
    dropped = windows[windows['hr_coverage'] < min_hr_coverage]
    if not dropped.empty:
        print(f"[WARN] Scartate {len(dropped)}/{n_windows} finestre con copertura cardiaca "
              f"< {min_hr_coverage:.0%}: window_id {dropped['window_id'].tolist()}")
    windows = windows[windows['hr_coverage'] >= min_hr_coverage]

    print(f"[Align] Stop video a {t_stop_vid:.2f}s = {hr_stop_ts} | t=0 video = {video_zero_ts} | "
          f"{len(windows)} finestre valide da {window_sec}s")

    cols = ['window_id', 'video_idx', 'vid_start_sec', 'vid_end_sec', 'ts_start', 'ts_end',
            'mu_hr', 'rmssd', 'pnn50', 'hr_coverage']
    return windows[cols].sort_values('ts_start').reset_index(drop=True)


if __name__ == "__main__":
    test_videos = [os.path.join(PROJECT_ROOT, "data", "video", v)
                   for v in ("camera_A000.mp4", "camera_A001.mp4")]
    test_cardiac_dir = os.path.join(PROJECT_ROOT, "data", "cardiac")

    df_windows = build_aligned_windows(
        video_paths=test_videos,
        stop_video_index=1,
        stop_time_in_segment=120.0,
        cardiac_path=test_cardiac_dir,
        window_sec=20,
    )
    print(df_windows.head())
    out = os.path.join(PROJECT_ROOT, "data", "aligned_data.csv")
    df_windows.to_csv(out, index=False)
    print(f"Salvato {out}")