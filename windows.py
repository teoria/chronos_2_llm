# Rolling backtest windows: last N_WIN days (test day included), each forecast from all prior data.
import pandas as pd

N_WIN, H = 30, 24
COVS = ["Ampirion Load Forecast", "PV+Wind Forecast"]
full = pd.concat([pd.read_parquet("data/train.parquet"), pd.read_parquet("data/test.parquet")], ignore_index=True)
FIRST = len(full) - H * N_WIN  # row where window 0 starts

def windows():
    for w in range(N_WIN):
        start = FIRST + H * w
        yield w, full.iloc[:start], full.iloc[start:start + H]
