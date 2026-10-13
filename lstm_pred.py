# LSTM encoder over the last 168h of [target, covs] + future covs -> next 24h.
# Trained once on the year before the backtest starts, then applied to every window (no leakage).
import numpy as np
import pandas as pd
import torch
from torch import nn
from windows import windows, full, FIRST, COVS, H

torch.manual_seed(0)
L = 168
feats = ["target"] + COVS
hist = full.iloc[:FIRST].tail(24 * 365)
mu, sd = hist[feats].mean(), hist[feats].std()
scale = lambda df: ((df[feats] - mu) / sd).to_numpy(np.float32)
Z = scale(hist)
idx = range(len(Z) - L - H + 1)
X = torch.tensor(np.stack([Z[i:i + L] for i in idx]))
F = torch.tensor(np.stack([Z[i + L:i + L + H, 1:] for i in idx]))
Y = torch.tensor(np.stack([Z[i + L:i + L + H, 0] for i in idx]))

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(len(feats), 64, batch_first=True)
        self.head = nn.Linear(64 + H * len(COVS), H)
    def forward(self, x, f):
        _, (h, _) = self.lstm(x)
        return self.head(torch.cat([h[-1], f.flatten(1)], 1))

net = Net()
opt = torch.optim.Adam(net.parameters(), lr=1e-3)
for epoch in range(15):
    for b in torch.randperm(len(X)).split(256):
        opt.zero_grad()
        loss = nn.functional.mse_loss(net(X[b], F[b]), Y[b])
        loss.backward(); opt.step()
    print(f"epoch {epoch} loss {loss.item():.4f}")

net.eval()
out = []
with torch.no_grad():
    for w, ctx, tgt in windows():
        x = torch.tensor(scale(ctx.tail(L)))[None]
        f = torch.tensor(scale(tgt.assign(target=0.0))[:, 1:])[None]
        out.append(net(x, f)[0].numpy() * sd["target"] + mu["target"])
pd.DataFrame({"lstm": np.concatenate(out)}).to_parquet("data/lstm_pred.parquet")
