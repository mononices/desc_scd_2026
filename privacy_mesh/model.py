import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from privacy_mesh.data import MODEL_COLUMNS


class RiskNet(nn.Module):
    def __init__(self, in_dim=len(MODEL_COLUMNS), hidden=192):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def logits(model, x_np, batch=1024):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x_np), batch):
            xb = torch.from_numpy(x_np[i:i + batch])
            out.append(model(xb).numpy())
    return np.concatenate(out)


def evaluate(model, x_np, y_np, batch=1024):
    z = logits(model, x_np, batch)
    p = 1.0 / (1.0 + np.exp(-z))
    pred = (p >= 0.5).astype(int)
    acc = accuracy_score(y_np, pred)
    auc = roc_auc_score(y_np, p)
    return acc, auc


def train_central(x_train, y_train, x_test, y_test, epochs=35, batch=256, lr=0.05,
                  seed=0, progress=None, eval_every=1):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = RiskNet()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    loss_fn = nn.BCEWithLogitsLoss()
    ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    loader = DataLoader(ds, batch_size=batch, shuffle=True)
    history = []
    total = epochs * max(1, len(loader))
    done = 0
    for ep in range(epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb.float())
            loss.backward()
            opt.step()
            done += 1
            if progress and done % max(1, total // 20) == 0:
                progress(done / total, f"centralised training epoch {ep + 1}/{epochs}")
        if (ep + 1) % eval_every == 0 or ep == epochs - 1:
            acc, auc = evaluate(model, x_test, y_test)
            history.append([ep + 1, float(acc), float(auc)])
    return model, history
