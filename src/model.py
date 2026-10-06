import torch
from torch import nn


class Net(nn.Module):
    def __init__(self, n_sensors: int):
        super().__init__()
        self.lstm = nn.LSTM(n_sensors, 64, num_layers=2, dropout=0.2, batch_first=True)
        self.head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1]).squeeze(-1)


def fit(x, y, xv, yv, seed: int, epochs=40, patience=8) -> Net:
    """Train on target scaled by 125, keep the weights with the best validation loss."""
    torch.manual_seed(seed)
    x, xv = torch.tensor(x, dtype=torch.float32), torch.tensor(xv, dtype=torch.float32)
    y, yv = torch.tensor(y / 125, dtype=torch.float32), torch.tensor(yv / 125, dtype=torch.float32)
    net = Net(x.shape[2])
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    best, best_state, stale = float("inf"), None, 0
    for epoch in range(epochs):
        net.train()
        for i in torch.randperm(len(x)).split(256):
            opt.zero_grad()
            nn.functional.mse_loss(net(x[i]), y[i]).backward()
            opt.step()
        loss = val_loss(net, xv, yv)
        print(f"  seed {seed} epoch {epoch + 1} val {loss:.5f}")
        if loss < best:
            best, stale = loss, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            stale += 1
            if stale == patience:
                break
    net.load_state_dict(best_state)
    return net.eval()


@torch.no_grad()
def val_loss(net, x, y) -> float:
    net.eval()
    return float(nn.functional.mse_loss(torch.cat([net(b) for b in x.split(4096)]), y))


@torch.no_grad()
def predict(net, x):
    x = torch.tensor(x, dtype=torch.float32)
    return torch.cat([net(b) for b in x.split(4096)]).numpy() * 125
