"""Differentiable clearance penalty on predicted endpoints, for the clearance-aware CFM baseline.

Obstacles come from the conditioning vector (feat = MAX_OBS x [kind, cx, cy, hw, hh], mask). Violation depth at a
point for clearance c matches Env.collision_mask(margin=c): circle -> r + c - |p - centre|; box (inflated
rectangle, square corners) -> min(hw + c - |dx|, hh + c - |dy|) when both are positive.
Points: every waypoint plus start/goal, with each segment subdivided into `sub` points.
"""
import torch


def endpoint_penalty(start, goal, max_obs, c, weight, sub=4, t_power=2.0):
    start = torch.tensor(start, dtype=torch.float32)
    goal = torch.tensor(goal, dtype=torch.float32)

    def aux(xt, t, v, cond):
        B = len(xt)
        x1 = xt + (1 - t)[:, None] * v                                   # predicted endpoint
        W = torch.cat([start.expand(B, 1, 2), x1.view(B, -1, 2), goal.expand(B, 1, 2)], 1)   # (B, T, 2)
        a = torch.linspace(0, 1, sub + 1)[:-1]
        P = (W[:, :-1, None] * (1 - a)[None, None, :, None] + W[:, 1:, None] * a[None, None, :, None]).reshape(B, -1, 2)
        feat = cond[:, : 5 * max_obs].view(B, max_obs, 5)
        mask = cond[:, 5 * max_obs: 6 * max_obs] > 0.5
        kind, cx, cy, hw, hh = (feat[..., k][:, None] for k in range(5))  # (B, 1, M)
        dx = P[..., 0:1] - cx
        dy = P[..., 1:2] - cy
        circ = hw + c - torch.sqrt(dx ** 2 + dy ** 2 + 1e-12)
        box = torch.minimum(hw + c - dx.abs(), hh + c - dy.abs())
        depth = torch.where(kind > 0.5, box, circ).clamp(min=0) * mask[:, None, :]
        per = depth.pow(2).sum((1, 2))
        return weight * (t.pow(t_power) * per).mean()
    return aux
