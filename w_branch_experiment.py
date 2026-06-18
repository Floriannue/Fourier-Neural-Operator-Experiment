"""
Code and data inspired from https://neuraloperator.github.io/dev/theory_guide/fno.html

w_branch_experiment.py: does the local W branch in FNO matter?

Tests three variants:
  - Full FNO : Fourier layer K  +  pointwise W  (paper default)
  - No W     : Fourier layer K  only
  - No FFT   : pointwise W  only

  burgers  : 1D Burgers (periodic BC)
             data: burgers_data_R10.mat
             keys: a (2048, 8192), u (2048, 8192)

  darcy    : 2D Darcy flow (non-periodic BC)
             data: piececonst_r241_N1024_smooth1.mat  (train)
                   piececonst_r241_N1024_smooth2.mat  (test)
             keys: coeff (1024, 241, 241), sol (1024, 241, 241)
"""

import os, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.io

#use full cpu if gpu not supported
torch.set_num_threads(os.cpu_count())

PROBLEM = 'darcy'  # <-- 'burgers' or 'darcy'

#config per problem
CFG = {
    'burgers': dict(
        data_train='burgers_data_R10.mat',
        data_test=None,
        N=1024,
        n_train=1000,
        n_test=200,
        modes=16,
        width=32,
        depth=4,
        epochs=300,
        lr=1e-3,
        batch=200,
    ),
    'darcy': dict(
        data_train='piececonst_r241_N1024_smooth1.mat',
        data_test='piececonst_r241_N1024_smooth2.mat',
        N=85,
        n_train=1000,
        n_test=100,
        modes=12,
        width=32,
        depth=4,
        epochs=150,
        lr=1e-3,
        batch=50,
    ),
}

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# torch.compile disabled: does not support complex ops (rfft)
USE_COMPILE = False


########################
#  1-D  (Burgers)
########################

class SpectralConv1d(nn.Module):
    def __init__(self, c_in, c_out, modes):
        super().__init__()
        self.modes = modes
        scale = 1.0 / (c_in * c_out)
        self.W = nn.Parameter(scale * torch.rand(c_in, c_out, modes, dtype=torch.cfloat))

    def forward(self, x):
        B, C, Nx = x.shape
        xf = torch.fft.rfft(x)
        out_ft = torch.zeros(B, self.W.shape[1], Nx // 2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[..., :self.modes] = torch.einsum('bim,iom->bom', xf[..., :self.modes], self.W)
        return torch.fft.irfft(out_ft, n=Nx)


class FNOBlock1d(nn.Module):
    def __init__(self, ch, modes, use_W, use_fft):
        super().__init__()
        assert use_W or use_fft
        self.use_fft = use_fft
        self.use_W = use_W
        if use_fft:
            self.K = SpectralConv1d(ch, ch, modes)
        if use_W:
            self.W = nn.Conv1d(ch, ch, 1)

    def forward(self, x):
        out = 0.0
        if self.use_fft:
            out = out + self.K(x)
        if self.use_W:
            out = out + self.W(x)
        return F.gelu(out)


class FNO1d(nn.Module):
    def __init__(self, modes, width, depth, use_W, use_fft):
        super().__init__()
        self.lift = nn.Linear(2, width)
        self.blocks = nn.ModuleList([FNOBlock1d(width, modes, use_W, use_fft) for _ in range(depth)])
        self.proj = nn.Sequential(nn.Linear(width, 128), nn.GELU(), nn.Linear(128, 1))

    def forward(self, a, grid):
        x = self.lift(torch.stack([a, grid], -1)).permute(0, 2, 1)
        for blk in self.blocks:
            x = blk(x)
        return self.proj(x.permute(0, 2, 1)).squeeze(-1)


####################
#  2-D  (Darcy)
####################

class SpectralConv2d(nn.Module):
    def __init__(self, c_in, c_out, modes):
        super().__init__()
        self.modes = modes
        scale = 1.0 / (c_in * c_out)
        self.W = nn.Parameter(scale * torch.rand(c_in, c_out, modes, modes, dtype=torch.cfloat))

    def forward(self, x):  # (B, C, H, W)
        B, C, H, Wx = x.shape
        m = self.modes
        xf = torch.fft.rfft2(x)
        out_ft = torch.zeros(B, self.W.shape[1], H, Wx // 2 + 1,dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :m, :m] = torch.einsum('bixy,ioxy->boxy', xf[:, :, :m, :m], self.W)
        return torch.fft.irfft2(out_ft, s=(H, Wx))


class FNOBlock2d(nn.Module):
    def __init__(self, ch, modes, use_W, use_fft):
        super().__init__()
        assert use_W or use_fft
        self.use_fft = use_fft
        self.use_W = use_W
        if use_fft:
            self.K = SpectralConv2d(ch, ch, modes)
        if use_W:
            self.W = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        out = 0.0
        if self.use_fft:
            out = out + self.K(x)
        if self.use_W:
            out = out + self.W(x)
        return F.gelu(out)


class FNO2d(nn.Module):
    def __init__(self, modes, width, depth, use_W, use_fft):
        super().__init__()
        self.lift = nn.Linear(3, width)
        self.blocks = nn.ModuleList([FNOBlock2d(width, modes, use_W, use_fft) for _ in range(depth)])
        self.proj = nn.Sequential(nn.Linear(width, 128), nn.GELU(), nn.Linear(128, 1))

    def forward(self, a, grid):
        x = self.lift(torch.cat([a.unsqueeze(-1), grid], -1)).permute(0, 3, 1, 2)
        for blk in self.blocks:
            x = blk(x)
        return self.proj(x.permute(0, 2, 3, 1)).squeeze(-1)



def _mat_keys(raw):
    keys = [k for k in raw.keys() if not k.startswith('_')]
    print(f"  .mat keys: {keys}")
    return keys


def load_burgers(cfg):
    raw = scipy.io.loadmat(cfg['data_train'])
    _mat_keys(raw)
    N, n_tr, n_te = cfg['N'], cfg['n_train'], cfg['n_test']

    a_np = raw['a'].astype(np.float32)
    u_np = raw['u'].astype(np.float32)
    step = a_np.shape[1] // N
    a_np = a_np[:, ::step][:, :N]
    u_np = u_np[:, ::step][:, :N]

    a = torch.tensor(a_np)
    u = torch.tensor(u_np)
    grid = torch.linspace(0, 1, N).unsqueeze(0).expand(len(a), -1)

    tr = (a[:n_tr], u[:n_tr], grid[:n_tr])
    te = (a[n_tr:n_tr + n_te], u[n_tr:n_tr + n_te], grid[n_tr:n_tr + n_te])
    return tr, te


def load_darcy(cfg):
    N, n_tr, n_te = cfg['N'], cfg['n_train'], cfg['n_test']

    def _load(path, n):
        raw = scipy.io.loadmat(path)
        _mat_keys(raw)
        a_key = 'coeff' if 'coeff' in raw else 'a'
        u_key = 'sol' if 'sol' in raw else 'u'
        a_np = raw[a_key][:n].astype(np.float32)
        u_np = raw[u_key][:n].astype(np.float32)
        step = a_np.shape[1] // N
        a_np = a_np[:, ::step, ::step][:, :N, :N]
        u_np = u_np[:, ::step, ::step][:, :N, :N]
        return torch.tensor(a_np), torch.tensor(u_np)

    a_tr, u_tr = _load(cfg['data_train'], n_tr)
    a_te, u_te = _load(cfg['data_test'], n_te)

    lin = torch.linspace(0, 1, N)
    gx, gy = torch.meshgrid(lin, lin, indexing='ij')
    grid = torch.stack([gx, gy], -1)
    g_tr = grid.unsqueeze(0).expand(n_tr, -1, -1, -1)
    g_te = grid.unsqueeze(0).expand(n_te, -1, -1, -1)

    return (a_tr, u_tr, g_tr), (a_te, u_te, g_te)



def rel_l2_1d(pred, target):
    return (torch.norm(pred - target, dim=1) /
            torch.norm(target, dim=1)).mean().item()


def rel_l2_2d(pred, target):
    return (torch.norm(pred.flatten(1) - target.flatten(1), dim=1) /
            torch.norm(target.flatten(1), dim=1)).mean().item()


def run(name, use_W, use_fft, tr_data, te_data, cfg, is2d):
    a_tr, u_tr, g_tr = tr_data
    a_te, u_te, g_te = te_data

    n_tr = cfg['n_train']
    epochs = cfg['epochs']
    batch = cfg['batch']
    rel_l2 = rel_l2_2d if is2d else rel_l2_1d

    if is2d:
        model = FNO2d(cfg['modes'], cfg['width'], cfg['depth'],
                      use_W, use_fft).to(DEVICE)
    else:
        model = FNO1d(cfg['modes'], cfg['width'], cfg['depth'],
                      use_W, use_fft).to(DEVICE)

    if USE_COMPILE:
        model = torch.compile(model)

    opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=100, gamma=0.5)

    idx = torch.arange(n_tr, device=DEVICE)
    t0 = time.time()

    for ep in range(1, epochs + 1):
        model.train()
        perm = idx[torch.randperm(n_tr, device=DEVICE)]
        for i in range(0, n_tr, batch):
            b = perm[i:i + batch]
            opt.zero_grad()
            pred = model(a_tr[b], g_tr[b])
            loss = F.mse_loss(pred, u_tr[b])
            loss.backward()
            opt.step()
        sched.step()

        if ep % 20 == 0:
            model.eval()
            with torch.inference_mode():
                err = rel_l2(model(a_te, g_te), u_te)
            elapsed = time.time() - t0
            print(f"  [{name}] ep {ep}  rel-L2 = {err}  ({elapsed}s)")

    model.eval()
    with torch.inference_mode():
        err = rel_l2(model(a_te, g_te), u_te)
    print(f"\n  >>> {name}  final rel-L2 = {err:.4f}  "
          f"(total {time.time() - t0:.0f}s)\n")
    return err


if __name__ == '__main__':
    cfg = CFG[PROBLEM]
    is2d = (PROBLEM == 'darcy')

    print(f"Problem: {PROBLEM}")
    print(f"Device: {DEVICE}  ({os.cpu_count()} CPU threads)")
    print(f"Compile: {USE_COMPILE}  (torch {torch.__version__})")
    print(f"N={cfg['N']}, width={cfg['width']}, modes={cfg['modes']}, "
          f"depth={cfg['depth']}, epochs={cfg['epochs']}, "
          f"batch={cfg['batch']}\n")

    print("Loading data...")
    if is2d:
        tr_data, te_data = load_darcy(cfg)
    else:
        tr_data, te_data = load_burgers(cfg)

    # move everything to device once — shared across all three runs
    tr_data = tuple(t.to(DEVICE) for t in tr_data)
    te_data = tuple(t.to(DEVICE) for t in te_data)

    print(f"Loaded: train={cfg['n_train']}, test={cfg['n_test']}, "
          f"resolution={cfg['N']}{'x' + str(cfg['N']) if is2d else ''}\n")

    results = {}
    for name, use_W, use_fft in [
        ('Full FNO', True, True),
        ('No W', False, True),
        ('No FFT', True, False),
    ]:
        results[name] = run(name, use_W, use_fft, tr_data, te_data, cfg, is2d)

    print("=" * 50)
    print(f"{PROBLEM.capitalize()} summary")
    print("=" * 50)
    base = results['Full FNO']
    for name, err in results.items():
        print(f"  {name:<14s}  {err:.4f}   ({err / base:.2f}x baseline)")
