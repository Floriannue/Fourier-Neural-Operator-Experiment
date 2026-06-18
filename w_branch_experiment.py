"""
w_branch_experiment.py: does the local W branch in FNO matter?

Tests three variants:
  - Full FNO : Fourier layer K  +  pointwise W  (paper default)
  - No W     : Fourier layer K  only
  - No FFT   : pointwise W  only
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.io

# Config
DATA_PATH = "burgers_data_R10.mat"
N = 1024  # sub-sample resolution (original: 8192)
N_TRAIN = 1000
N_TEST = 200
MODES = 16  # k_max
WIDTH = 32  # d_v
DEPTH = 4
EPOCHS = 300
LR = 1e-3
BATCH = 50
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class SpectralConv1d(nn.Module):
    def __init__(self, c_in, c_out, modes):
        super().__init__()
        self.modes = modes
        scale = 1.0 / (c_in * c_out)
        self.W = nn.Parameter(
            scale * torch.rand(c_in, c_out, modes, dtype=torch.cfloat))

    def forward(self, x):
        B, C, Nx = x.shape
        x_ft = torch.fft.rfft(x)
        out_ft = torch.zeros(B, self.W.shape[1], Nx // 2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[..., :self.modes] = torch.einsum('bim,iom->bom', x_ft[..., :self.modes], self.W)
        return torch.fft.irfft(out_ft, n=Nx)


class FNOBlock(nn.Module):
    """y = σ( K(x) + W(x) )"""

    def __init__(self, channels, modes, use_W=True, use_fft=True):
        super().__init__()
        assert use_W or use_fft
        self.use_fft = use_fft
        self.use_W = use_W
        if use_fft:
            self.K = SpectralConv1d(channels, channels, modes)
        if use_W:
            self.W = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x):
        out = 0.0
        if self.use_fft:
            out = out + self.K(x)
        if self.use_W:
            out = out + self.W(x)
        return F.gelu(out)


class FNO1d(nn.Module):
    def __init__(self, modes=MODES, width=WIDTH, depth=DEPTH, use_W=True, use_fft=True):
        super().__init__()
        self.lift = nn.Linear(2, width)  # (a(x), x) -> d_v channels
        self.blocks = nn.ModuleList([FNOBlock(width, modes, use_W, use_fft) for _ in range(depth)])
        self.proj = nn.Sequential(nn.Linear(width, 128), nn.GELU(), nn.Linear(128, 1))

    def forward(self, a, grid):  # (B, N)
        x = self.lift(torch.stack([a, grid], -1))  # (B, N, width)
        x = x.permute(0, 2, 1)  # (B, width, N)
        for blk in self.blocks:
            x = blk(x)
        return self.proj(x.permute(0, 2, 1)).squeeze(-1)


def load_data():
    raw = scipy.io.loadmat(DATA_PATH)
    a_np = raw['a'].astype(np.float32)  # (2048, 8192)
    u_np = raw['u'].astype(np.float32)

    #sub-sample uniformly in space
    step = a_np.shape[1] // N
    a_np = a_np[:, ::step][:, :N]
    u_np = u_np[:, ::step][:, :N]

    a = torch.tensor(a_np)
    u = torch.tensor(u_np)
    grid = torch.linspace(0, 1, N).unsqueeze(0).expand(len(a), -1)

    tr = (a[:N_TRAIN], u[:N_TRAIN], grid[:N_TRAIN])
    te = (a[N_TRAIN:N_TRAIN + N_TEST], u[N_TRAIN:N_TRAIN + N_TEST],
          grid[N_TRAIN:N_TRAIN + N_TEST])
    return tr, te


def rel_l2(pred, target):
    return (torch.norm(pred - target, dim=1) /
            torch.norm(target, dim=1)).mean().item()


#train + evaluate
def run(name, use_W, use_fft, tr_data, te_data):
    a_tr, u_tr, g_tr = [t.to(DEVICE) for t in tr_data]
    a_te, u_te, g_te = [t.to(DEVICE) for t in te_data]

    model = FNO1d(use_W=use_W, use_fft=use_fft).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=100, gamma=0.5)

    idx = torch.arange(N_TRAIN)
    for ep in range(1, EPOCHS + 1):
        model.train()
        perm = idx[torch.randperm(N_TRAIN)]
        for i in range(0, N_TRAIN, BATCH):
            b = perm[i:i + BATCH]
            opt.zero_grad()
            pred = model(a_tr[b], g_tr[b])
            loss = F.mse_loss(pred, u_tr[b])
            loss.backward()
            opt.step()
        sched.step()
        if ep % 20 == 0:
            model.eval()
            with torch.no_grad():
                err = rel_l2(model(a_te, g_te), u_te)
            print(f"[{name}] ep {ep}  rel-L2 = {err}")

    model.eval()
    with torch.no_grad():
        err = rel_l2(model(a_te, g_te), u_te)
    print(f"\n>>> {name}  final rel-L2 = {err}\n")
    return err


if __name__ == '__main__':
    print(f"Device: {DEVICE}")
    print(f"Data: {DATA_PATH}")
    print(f"N={N}, width={WIDTH}, modes={MODES}, depth={DEPTH}, epochs={EPOCHS}\n")

    tr_data, te_data = load_data()
    print(f"Loaded: train={N_TRAIN}, test={N_TEST}, resolution={N}\n")

    results = {}
    for name, use_W, use_fft in [
        ('Full FNO', True, True),
        ('No W', False, True),
        ('No FFT', True, False),
    ]:
        results[name] = run(name, use_W, use_fft, tr_data, te_data)

    print("=" * 45)
    print("Burgers summary")
    print("=" * 45)
    base = results['Full FNO']
    for name, err in results.items():
        print(f"  {name}  {err:.4f}   ({err / base:.2f}x baseline)")
