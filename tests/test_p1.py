import torch

from topoflow.p1 import ObstacleAttnField


def _permute_cond(cond, perm, M=3):
    B = cond.shape[0]
    feat = cond[:, : 5 * M].view(B, M, 5)[:, perm].reshape(B, -1)
    mask = cond[:, 5 * M : 6 * M][:, perm]
    code = cond[:, 6 * M : 7 * M][:, perm]
    return torch.cat([feat, mask, code], 1)


def test_obstacle_order_invariance():
    torch.manual_seed(0)
    m = ObstacleAttnField(30, 3, d=32, depth=2).eval()
    x = torch.randn(4, 60)
    t = torch.rand(4)
    cond = torch.rand(4, 21)
    cond[:, 15:18] = torch.tensor([1.0, 1.0, 0.0])  # third slot empty
    a = m(x, t, cond)
    b = m(x, t, _permute_cond(cond, [2, 0, 1]))
    assert torch.allclose(a, b, atol=1e-5)


def test_empty_slots_are_ignored():
    torch.manual_seed(0)
    m = ObstacleAttnField(30, 3, d=32, depth=2).eval()
    x, t = torch.randn(2, 60), torch.rand(2)
    cond = torch.rand(2, 21)
    cond[:, 15:18] = torch.tensor([1.0, 0.0, 0.0])
    cond2 = cond.clone()
    cond2[:, 5:15] = torch.rand(2, 10)   # garbage in masked slots
    cond2[:, 19:21] = 1.0
    assert torch.allclose(m(x, t, cond), m(x, t, cond2), atol=1e-5)


def test_code_blind_variant_ignores_route_bits():
    torch.manual_seed(0)
    m = ObstacleAttnField(30, 3, d=32, depth=2, use_code=False).eval()
    x, t = torch.randn(3, 60), torch.rand(3)
    cond = torch.rand(3, 21)
    cond2 = cond.clone()
    cond2[:, 18:21] = 1.0 - cond2[:, 18:21]
    assert torch.allclose(m(x, t, cond), m(x, t, cond2), atol=1e-6)
