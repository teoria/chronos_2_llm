# Rolling backtest over the last N_WIN days. Chronos/LSTM run in their own process (chronos_pred.py, lstm_pred.py):
# torch + xgboost/lightgbm/catboost in one process segfaults on macOS (duplicate libomp).
import os, subprocess, sys, warnings
import numpy as np
import pandas as pd
import scikit_posthocs as sp
from scipy.stats import friedmanchisquare
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from windows import windows, full, FIRST, COVS, N_WIN, H

warnings.filterwarnings("ignore")
y_true = full["target"].iloc[FIRST:].to_numpy().reshape(N_WIN, H)

def cached(name):
    path = f"data/{name}_pred.parquet"
    if not os.path.exists(path):
        subprocess.run([sys.executable, f"{name}_pred.py"], check=True)
    return pd.read_parquet(path)[name].to_numpy().reshape(N_WIN, H)

preds = {"Chronos-2": cached("chronos"), "Chronos-2+CoCo-LoRA": cached("chronos_lora"), "LSTM": cached("lstm")}

# Statistical models: refit per window on the last 4 weeks of context
# ponytail: fixed orders; swap for auto_arima if it matters
stat = {"ARIMAX(2,0,1)": [], "SARIMAX(1,0,1)(1,0,1,24)": [], "Holt-Winters": []}
for w, ctx, tgt in windows():
    hist = ctx.tail(24 * 28)
    y, ex, exf = hist["target"].to_numpy(), hist[COVS].to_numpy(), tgt[COVS].to_numpy()
    stat["ARIMAX(2,0,1)"].append(ARIMA(y, exog=ex, order=(2, 0, 1)).fit().forecast(H, exog=exf))
    stat["SARIMAX(1,0,1)(1,0,1,24)"].append(
        SARIMAX(y, exog=ex, order=(1, 0, 1), seasonal_order=(1, 0, 1, 24)).fit(disp=False).forecast(H, exog=exf))
    stat["Holt-Winters"].append(
        ExponentialSmoothing(y, trend="add", seasonal="add", seasonal_periods=24).fit().forecast(H))
    print(f"window {w} done", file=sys.stderr)
preds.update({k: np.array(v) for k, v in stat.items()})

# ML models: fit once on everything before the backtest, features = covs + calendar + lags (24h, 168h)
X = full[COVS].copy()
X["hour"] = full["timestamp"].dt.hour
X["dow"] = full["timestamp"].dt.dayofweek
X["lag24"] = full["target"].shift(24)
X["lag168"] = full["target"].shift(168)
Xtr, ytr, Xte = X.iloc[168:FIRST], full["target"].iloc[168:FIRST], X.iloc[FIRST:]
ml = {
    "RandomForest": RandomForestRegressor(n_estimators=300, random_state=0, n_jobs=-1),
    "XGBoost": XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, random_state=0),
    "LightGBM": LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=31, random_state=0, verbose=-1),
    "CatBoost": CatBoostRegressor(iterations=500, learning_rate=0.05, depth=6, random_seed=0, verbose=0),
    "MLP": make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=500, random_state=0)),
}
preds.update({k: m.fit(Xtr, ytr).predict(Xte).reshape(N_WIN, H) for k, m in ml.items()})

# Overall scores + Friedman/Nemenyi with windows as blocks
mae = pd.DataFrame({k: np.abs(p - y_true).mean(axis=1) for k, p in preds.items()})
rmse = {k: np.sqrt(((p - y_true) ** 2).mean()) for k, p in preds.items()}
ranks = mae.rank(axis=1).mean()
table = pd.DataFrame({"MAE": mae.mean(), "RMSE": pd.Series(rmse), "mean_rank": ranks}).sort_values("mean_rank")
best = ranks.idxmin()
table["nemenyi_p_vs_best"] = sp.posthoc_nemenyi_friedman(mae)[best]
chi2, p = friedmanchisquare(*[mae[c] for c in mae])
print(f"{N_WIN} windows x {H}h. Friedman chi2={chi2:.2f} p={p:.2e}\n")
print(table.round(3).to_string())

# Critical difference diagram (Nemenyi): models joined by a bar are not significantly different (alpha=0.05)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 4))
sp.critical_difference_diagram(ranks, sp.posthoc_nemenyi_friedman(mae))
plt.title(f"Nemenyi critical difference diagram, {N_WIN} daily windows, MAE rank (lower is better)")
plt.tight_layout()
plt.savefig("nemenyi.png", dpi=150)
print("saved nemenyi.png")

# Time series plot: actual target from START, forecasts for the test day only (last backtest window)
START, MODELS = "2017-12-10", ["SARIMAX(1,0,1)(1,0,1,24)", "Chronos-2", "LSTM"]
hist = full[full["timestamp"] >= START]
test_day = full.iloc[-H:]
plt.figure(figsize=(14, 5))
plt.plot(hist["timestamp"], hist["target"], color="black", lw=2, label="actual (target)")
for name in MODELS:
    plt.plot(test_day["timestamp"], preds[name][-1], "--", label=f"{name} (MAE {mae[name].iloc[-1]:.2f})")
plt.axvspan(test_day["timestamp"].iloc[0], test_day["timestamp"].iloc[-1], color="grey", alpha=0.1, label="test day")
plt.ylabel("electricity price"); plt.title(f"DE electricity price from {START}: actual vs 24h forecasts for the test day")
plt.legend(loc="upper left", fontsize=8); plt.tight_layout()
plt.savefig("forecast.png", dpi=150)
print("saved forecast.png")
