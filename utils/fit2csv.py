import os
import sys

import fitparse
import pandas as pd
from fitparse.utils import FitCRCError

allowed_fields = ['timestamp', 'hr', 'hrv', 'accx', 'accy', 'accz', 'exception', 'battery', 'spo2']
# Campi minimi perché una riga sia utile all'analisi cardiaca
required_fields = ['timestamp', 'hr', 'hrv']

TIMEZONE = 'Europe/Vienna'  # stesso fuso dell'Italia


def read_fit(fit_path: str) -> tuple[pd.DataFrame, Exception | None]:
    """
    Legge un file .fit in un DataFrame (timestamp tz-aware, ordinato).

    Ritorna (df, errore):
      - df pieno, errore None   -> file letto correttamente
      - df pieno, errore != None -> file troncato/danneggiato: tenute solo le righe lette prima dell'errore
      - df vuoto, errore != None -> file illeggibile
    """
    rows, error = [], None
    try:
        fitfile = fitparse.FitFile(fit_path, data_processor=fitparse.StandardUnitsDataProcessor())
        for message in fitfile.get_messages():
            row = {f.name: f.value for f in message.fields if f.name in allowed_fields}
            if all(row.get(k) is not None for k in required_fields):
                rows.append(row)
    except FitCRCError as e:
        # Il CRC si verifica a fine file, dopo aver letto tutti i record: i dati sono completi
        print(f"[WARN] {os.path.basename(fit_path)}: CRC non valido, dati comunque letti per intero ({e})")
    except Exception as e:
        error = e

    df = pd.DataFrame(rows, columns=allowed_fields)
    if not df.empty:
        # I timestamp FIT sono in UTC (naive): li rendiamo espliciti
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_convert(TIMEZONE)
        df = df.sort_values('timestamp').reset_index(drop=True)
    return df, error


def fit_to_csv(fit_path: str, csv_path: str) -> bool:
    """Converte un .fit in .csv. Ritorna False se il file è illeggibile."""
    df, error = read_fit(fit_path)
    if df.empty:
        print(f"[SKIP] {os.path.basename(fit_path)} illeggibile: {error}")
        return False
    if error is not None:
        print(f"[WARN] {os.path.basename(fit_path)} danneggiato, salvate {len(df)} righe valide: {error}")
    df.to_csv(csv_path, index=False)
    print(f"wrote {csv_path}")
    return True


def main(cardiac_dir: str = "data/cardiac"):
    fit_files = sorted(f for f in os.listdir(cardiac_dir) if f.lower().endswith('.fit'))
    if not fit_files:
        print(f"[WARN] Nessun file .fit trovato in {cardiac_dir}")
        return
    for f in fit_files:
        fit_to_csv(os.path.join(cardiac_dir, f), os.path.join(cardiac_dir, f[:-4] + '.csv'))
    print("finished conversions")


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else "data/cardiac")