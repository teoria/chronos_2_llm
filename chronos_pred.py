import numpy as np
import pandas as pd
from chronos.chronos2 import Chronos2Pipeline
from windows import windows

def run(pipeline, name):
    out = []
    for w, ctx, tgt in windows():
        pred = pipeline.predict_df(
            ctx, future_df=tgt.drop(columns="target"), prediction_length=len(tgt), quantile_levels=[0.5],
            id_column="id", timestamp_column="timestamp", target="target",
        )
        out.append(pred["0.5"].to_numpy())
        print(f"window {w} done")
    pd.DataFrame({name: np.concatenate(out)}).to_parquet(f"data/{name}_pred.parquet")

if __name__ == "__main__":
    run(Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="auto"), "chronos")
