import torch

def decision_hit_rate(pred, target, tol: float):
    """四维动作都满足 |pred-target| <= tol*|target| 记为命中"""
    eps = 1e-6
    rel_ok = (torch.abs(pred - target) <= tol * (torch.abs(target)+eps))
    hit = torch.all(rel_ok, dim=1).float().mean().item()
    return hit
