import math

import numpy as np
import torch
import torch.nn.functional as F
from opacus import PrivacyEngine
from torch.utils.data import DataLoader, TensorDataset

from privacy_mesh.model import RiskNet, evaluate
from privacy_mesh.privacy import calibrate_sigma


class _Client:
    def __init__(self, x, y, batch, lr, rounds, local_epochs, clip, seed, eps, delta, idx):
        self.n = len(x)
        loader_len = max(1, math.ceil(self.n / batch))
        self.sample_rate = 1.0 / loader_len
        self.steps_per_round = max(1, int(round(local_epochs * loader_len)))
        ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
        base_loader = DataLoader(
            ds,
            batch_size=batch,
            shuffle=eps is None,
            generator=torch.Generator().manual_seed(seed + idx) if eps is None else None,
        )
        self.model = RiskNet()
        self.state_model = self.model
        torch.manual_seed(seed + 1000 + idx)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        self.engine = None
        self.sigma = None
        if eps is not None:
            self.sigma = calibrate_sigma(
                eps, delta, self.sample_rate, self.steps_per_round * rounds
            )
            if self.sigma is None:
                raise ValueError(
                    f"cannot reach epsilon={eps:g} at delta={delta:g} for a client with "
                    f"{self.n} records over {self.steps_per_round * rounds} DP-SGD steps: "
                    "the required noise exceeds the calibration range. Raise epsilon, "
                    "reduce rounds/local_epochs, or use a larger cohort."
                )
            engine = PrivacyEngine()
            wrapped_model, wrapped_optimizer, dp_loader = engine.make_private(
                module=self.model,
                optimizer=self.optimizer,
                data_loader=base_loader,
                noise_multiplier=self.sigma,
                max_grad_norm=clip,
            )
            self.model = wrapped_model
            self.optimizer = wrapped_optimizer
            self.loader = dp_loader
            self.engine = engine
        else:
            self.loader = base_loader

    def local_steps(self, global_state):
        self.state_model.load_state_dict(global_state)
        self.optimizer.state.clear()
        self.model.train()
        done = 0
        if self.engine is None:
            epochs_done = 0
            while done < self.steps_per_round:
                for xb, yb in self.loader:
                    if done >= self.steps_per_round:
                        break
                    self._step(xb, yb)
                    done += 1
                epochs_done += 1
                if epochs_done > 50:
                    break
        else:
            it = iter(self.loader)
            while done < self.steps_per_round:
                try:
                    xb, yb = next(it)
                except StopIteration:
                    it = iter(self.loader)
                    xb, yb = next(it)
                if len(xb) == 0:
                    continue
                self._step(xb, yb)
                done += 1

    def _step(self, xb, yb):
        self.optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(self.model(xb), yb.float())
        loss.backward()
        self.optimizer.step()

    def report(self, delta):
        if self.engine is not None:
            return float(self.engine.get_epsilon(delta=delta))
        return None


def train_federated(entity_train, x_test, y_test, rounds=12, local_epochs=2, batch=256,
                    lr=0.05, eps=None, delta=1e-5, clip=1.0, seed=0, progress=None):
    torch.manual_seed(seed)
    clients = [
        _Client(x, y, batch, lr, rounds, local_epochs, clip, seed, eps, delta, i)
        for i, (x, y) in enumerate(entity_train)
    ]
    counts = np.array([float(c.n) for c in clients], dtype=np.float64)
    weights = counts / counts.sum()
    global_model = RiskNet()
    history = []
    for r in range(rounds):
        states = []
        for c in clients:
            c.local_steps(global_model.state_dict())
            states.append({k: v.detach().clone() for k, v in c.state_model.state_dict().items()})
        averaged = {
            k: sum(st[k] * w for st, w in zip(states, weights))
            for k in states[0]
        }
        global_model.load_state_dict(averaged)
        acc, auc = evaluate(global_model, x_test, y_test)
        history.append([r + 1, float(acc), float(auc)])
        if progress:
            progress((r + 1) / rounds, f"round {r + 1}/{rounds} done - test accuracy {acc:.3f}")
    epsilons = [c.report(delta) for c in clients]
    epsilons = [e for e in epsilons if e is not None]
    sigmas = [c.sigma for c in clients if c.sigma is not None]
    return {
        "model": global_model,
        "history": history,
        "eps_achieved": float(max(epsilons)) if epsilons else None,
        "eps_mean": float(np.mean(epsilons)) if epsilons else None,
        "eps_per_client": [float(e) for e in epsilons],
        "sigma": float(max(sigmas)) if sigmas else None,
        "sigma_per_client": [float(s) for s in sigmas],
        "client_sizes": [int(c.n) for c in clients],
    }
