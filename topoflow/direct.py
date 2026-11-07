"""DiRecT-style constrained sampling for rectified-flow samplers (independent baseline; research phase).

Re-implementation of the update rule of Giaretta, Li, Azizan, "DiRecT: Safe Diffusion-Based Planning via
Receding-Horizon Denoising" (arXiv 2606.15359), translated to rectified flow. NOT their code.

Per optimized step (t -> t + dt):
    xbar   = x_t + dt * v(x_t, t)                 uncontrolled proposal (Euler)
    x1hat  = xbar + (1 - t - dt) * v(xbar, t+dt)  clean estimate at the proposal (one extra field call)
    X*     = argmin_{X in S} ||X - x1hat||^2      (lambda = 0: pure projection)
    x_{t+dt} = xbar + (t + dt) * (X* - x1hat)     latent correction (coefficient of x1 in x_t is t)
The final step returns X* directly (Prop. 1 of the paper: feasible iff the last subproblem is solved).
Steps with t < T_OPT (default 0.5, the paper's "second half") are plain Euler. On the last step x1hat = xbar.

Deviations from the paper (stated in DIRECT_PLAN.md / results):
- IPOPT replaced by a batched PyTorch penalty solver written here (no external NLP solver).
- flow matching instead of diffusion; Tweedie estimate replaced by the rectified-flow clean estimate.

Constraint sets:
- "free": every densified point of the polyline (start, waypoints, goal) has g(p) >= c + tol, where
  g = |p - centre| - r for disks and max(|dx| - hw, |dy| - hh) for boxes (matches Env.collision_mask with margin c:
  disks inflated by c, boxes inflated to rectangles). Non-convex; class not enforced.
- "chart": the requested class's n = 60 chart (topoflow.chart). The clean estimate is resampled on the chart grid,
  clipped to [lo, hi] (exact Euclidean projection in chart coordinates), and the correction is mapped back to the
  base model's waypoints by linear interpolation in x. The returned path on the last step is the chart polyline.
"""
import time

import numpy as np
import torch

from . import chart
from .e2 import MAX_OBS, env_features, pad_code
from .tdfm import index_code


def obstacle_tensors(layouts, rows_layout):
    """Padded obstacle params per sample row: kind (B,M), centre (B,M,2), size (B,M,2), mask (B,M)."""
    L = len(layouts)
    kind = np.zeros((L, MAX_OBS)); cen = np.zeros((L, MAX_OBS, 2)); siz = np.ones((L, MAX_OBS, 2)); msk = np.zeros((L, MAX_OBS))
    for i, l in enumerate(layouts):
        for j, o in enumerate(l["env"].obstacles):
            kind[i, j] = 0.0 if o.kind == "circle" else 1.0
            cen[i, j] = (o.cx, o.cy)
            siz[i, j] = (o.r, o.r) if o.kind == "circle" else (o.hw, o.hh)
            msk[i, j] = 1.0
    idx = np.asarray(rows_layout)
    f = lambda a: torch.tensor(a[idx], dtype=torch.float64)  # noqa: E731
    return f(kind), f(cen), f(siz), f(msk)


def clearance_g(pts, kind, cen, siz, msk):
    """pts (B,P,2) -> min over obstacles of g (B,P). Masked obstacles ignored."""
    d = pts[:, :, None, :] - cen[:, None]                       # (B,P,M,2)
    circ = torch.sqrt((d ** 2).sum(-1) + 1e-18) - siz[:, None, :, 0]
    box = torch.maximum(d[..., 0].abs() - siz[:, None, :, 0], d[..., 1].abs() - siz[:, None, :, 1])
    g = torch.where(kind[:, None] > 0.5, box, circ)
    g = torch.where(msk[:, None] > 0.5, g, torch.full_like(g, 1e3))
    return g.min(-1).values


def densify(W, start, goal, sub):
    """W (B,T,2) waypoints, start/goal (B,2) -> densified points (B,(T+1)*sub+1,2)."""
    P = torch.cat([start[:, None], W, goal[:, None]], 1)
    a = torch.linspace(0, 1, sub + 1, dtype=W.dtype)[:-1]
    S = P[:, :-1, None] * (1 - a)[None, None, :, None] + P[:, 1:, None] * a[None, None, :, None]
    return torch.cat([S.reshape(len(W), -1, 2), P[:, -1:]], 1)


def project_free(Xt, start, goal, obs, c, iters=100, sub=16, tol=2e-3, mu0=1e2, mu1=1e6, lr=5e-3):
    """Penalty-continuation projection onto the clearance-c free set. Returns (X*, residual violation per row).

    minimise 0.5 ||X - Xt||^2 + 0.5 mu_k sum relu(c + tol - g)^2, mu_k geometric from mu0 to mu1, Adam steps.
    Residual = max over densified points of relu(c - g) (0 = feasible at c by this point check)."""
    X = Xt.clone().requires_grad_(True)
    opt = torch.optim.Adam([X], lr=lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: 1.0 - k / max(iters, 1))   # decay to 0: settle, not jitter
    mus = np.geomspace(mu0, mu1, max(iters, 1))
    for k in range(iters):
        g = clearance_g(densify(X, start, goal, sub), *obs)
        viol = torch.relu(c + tol - g)
        loss = 0.5 * ((X - Xt) ** 2).sum() + 0.5 * mus[k] * (viol ** 2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    with torch.no_grad():
        res = torch.relu(c - clearance_g(densify(X, start, goal, sub), *obs)).max(1).values
    return X.detach(), res


class ChartProj:
    """Projection onto the requested class's n=60 chart, in chart coordinates (see module doc)."""

    def __init__(self, layouts, req, c, n=60):
        self.lo, self.hi, self.xs, self.ok = [], [], [], []
        cache = {}
        for i, code in req:
            k = (i, code)
            if k not in cache:
                env = layouts[i]["env"]
                lo, hi, ok = chart.bounds(env, code, c, n)
                cache[k] = (lo, hi, chart.grid(env, n, c), ok)
            lo, hi, xs, ok = cache[k]
            self.lo.append(lo); self.hi.append(hi); self.xs.append(xs); self.ok.append(ok)
        self.lo, self.hi, self.xs = (torch.tensor(np.stack(a)) for a in (self.lo, self.hi, self.xs))
        self.ok = np.array(self.ok)

    def __call__(self, W):
        """W (B,T,2) -> (chart polyline y on xs (B,n), delta mapped back to W (B,T,2))."""
        x, y = W[..., 0].numpy(), W[..., 1].numpy()
        xs = self.xs.numpy()
        order = np.argsort(x, 1)
        xo, yo = np.take_along_axis(x, order, 1), np.take_along_axis(y, order, 1)
        yc = np.stack([np.interp(xs[b], xo[b], yo[b]) for b in range(len(W))])
        m, w = (self.lo + self.hi) / 2, (self.hi - self.lo) / 2 * (1 - chart.EPS)
        yp = np.clip(yc, (m - w).numpy(), (m + w).numpy())
        dy = np.stack([np.interp(x[b], xs[b], yp[b] - yc[b]) for b in range(len(W))])
        delta = torch.zeros_like(W)
        delta[..., 1] = torch.tensor(dy, dtype=W.dtype)
        return torch.tensor(yp, dtype=W.dtype), delta


@torch.no_grad()
def field(model, x, t, cond):
    return model(x.float(), torch.full((len(x),), float(t)), cond).double()


def sample(model, layouts, req, cond, x0, K, mode, c, iters=100, t_opt=0.5, sub=16, batch=1024):
    """mode in {"none", "final", "free", "chart"}. req: list of (layout index, code). x0: (B, 60) noise.
    Returns (paths list of (T+2,2) arrays, info dict with timings and solver residuals)."""
    B = len(req)
    start = torch.tensor(np.stack([layouts[i]["env"].start for i, _ in req]), dtype=torch.float64)
    goal = torch.tensor(np.stack([layouts[i]["env"].goal for i, _ in req]), dtype=torch.float64)
    obs = obstacle_tensors(layouts, [i for i, _ in req])
    cp = ChartProj(layouts, req, c) if mode == "chart" else None
    out, res_all, t_field, t_proj, chart_y = [], [], 0.0, 0.0, []
    for s in range(0, B, batch):
        sl = slice(s, s + batch)
        x = x0[sl].double().clone()
        cb = cond[sl]
        ob = tuple(o[sl] for o in obs)
        last_res = torch.zeros(len(x), dtype=torch.float64)
        yfinal = None
        for k in range(K):
            t, dt = k / K, 1.0 / K
            a = time.perf_counter()
            v = field(model, x, t, cb)
            xbar = x + dt * v
            optimize = mode in ("free", "chart") and t >= t_opt - 1e-12 or (mode == "final" and k == K - 1) \
                or (mode in ("free", "chart") and k == K - 1)
            if not optimize:
                t_field += time.perf_counter() - a
                x = xbar
                continue
            if k < K - 1:
                x1hat = xbar + (1 - t - dt) * field(model, xbar, t + dt, cb)
            else:
                x1hat = xbar
            t_field += time.perf_counter() - a
            a = time.perf_counter()
            W = x1hat.reshape(len(x), -1, 2)
            if mode in ("free", "final"):
                Ws, res = project_free(W, start[sl], goal[sl], ob, c, iters=iters, sub=sub)
                delta = Ws - W
                last_res = res
            else:
                cp_b = ChartProj.__new__(ChartProj)
                cp_b.lo, cp_b.hi, cp_b.xs = cp.lo[sl], cp.hi[sl], cp.xs[sl]
                yfinal, delta = cp_b(W)
            t_proj += time.perf_counter() - a
            if k < K - 1:
                x = xbar + (t + dt) * delta.reshape(len(x), -1)
            else:
                x = (W + delta).reshape(len(x), -1)
        res_all.append(last_res)
        W = x.reshape(len(x), -1, 2).numpy()
        for j in range(len(W)):
            i = req[s + j][0]
            if mode == "chart":
                env = layouts[i]["env"]
                out.append(chart.to_path(env, yfinal[j].numpy(), c))   # final output = chart polyline (Prop. 1)
            else:
                out.append(np.concatenate([[layouts[i]["env"].start], W[j], [layouts[i]["env"].goal]]))
    res = torch.cat(res_all).numpy()
    info = dict(t_field=t_field, t_proj=t_proj, residual_mean=float(res.mean()), residual_max=float(res.max()),
                residual_pos_frac=float((res > 0).mean()),
                chart_infeasible=float((~cp.ok).mean()) if cp is not None else 0.0)
    return out, info


def build_cond(layouts, req, use_code, sg=False):
    conds = []
    for i, code in req:
        lay = layouts[i]
        f, m = env_features(lay["env"])
        extra = [lay["env"].start[1:2], lay["env"].goal[1:2]] if sg else []
        conds.append(np.concatenate([f, m, pad_code(code), *extra]))
    return torch.tensor(np.array(conds), dtype=torch.float32)


__all__ = ["sample", "build_cond", "project_free", "ChartProj", "clearance_g", "index_code"]
