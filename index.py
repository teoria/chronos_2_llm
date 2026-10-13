import pandas as pd
from chronos.chronos2 import Chronos2Model, Chronos2Pipeline
from peft import PeftModel

# CPU: float32, no device_map (bf16 + device_map="auto" aborts on this machine)
base = Chronos2Model.from_pretrained("amazon/chronos-2")
model = PeftModel.from_pretrained(base, "SWARM-Lab/CoCo-Chronos2-LoRA").merge_and_unload().eval()
pipeline = Chronos2Pipeline(model)
# Load historical target values and past values of covariates
context_df = pd.read_parquet("data/train.parquet")

# (Optional) Load future values of covariates
future_df = pd.read_parquet("data/test.parquet").drop(columns="target")

# Generate predictions with covariates
pred_df = pipeline.predict_df(
    context_df,
    future_df=future_df,
    prediction_length=24,  # Number of steps to forecast
    quantile_levels=[0.1, 0.5, 0.9],  # Quantiles for probabilistic forecast
    id_column="id",  # Column identifying different time series
    timestamp_column="timestamp",  # Column with datetime information   
    target="target",  # Column(s) with time series values to predict
)

print(pred_df.head())