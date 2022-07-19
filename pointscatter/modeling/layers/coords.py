import torch
from scipy.optimize import linear_sum_assignment
import numpy as np
import warnings
warnings.filterwarnings("ignore")


def get_meshgrid(shape, device):
    assert len(shape) <= 3
    ns = [torch.arange(0, s, dtype=torch.float, device=device) + 0.5 for s in shape]
    grids = torch.meshgrid(*ns)
    return torch.stack(grids, dim=-1)


def get_centered_meshgrid(shape, device):
    meshgrid = get_meshgrid(shape, device)
    center = torch.tensor(shape, dtype=torch.float32, device=device) / 2
    meshgrid = meshgrid - center
    return meshgrid


def batched_greedy_assignment(cost):
    """
    The shape of cost is (*, N, K)
    Recommended setting: N=num_pred, K=num_gt
    return: shape of (*, min(N, K)), LongTensor
    """
    cost = cost.clone()
    batch_shape = cost.shape[:-2]
    N, K = cost.shape[-2], cost.shape[-1]
    points = min(N, K)
    assignment_shape = (*batch_shape, points)
    assignment = - torch.ones(size=assignment_shape, dtype=torch.long, device=cost.device)

    for i in range(points):
        cur_min_idx = torch.argmin(cost[..., i], dim=-1)
        assignment[..., i] = cur_min_idx
        index = torch.repeat_interleave(cur_min_idx[..., None, None], repeats=K, dim=-1)
        cost.scatter_(dim=-2, index=index, value=float('inf'))

    return assignment


def Hungarian_assignment(cost):
    """
    The shape of cost is (*, N, K)
    Recommended setting: N=num_pred, K=num_gt
    return: shape of (*, min(N, K)), LongTensor
    """
    cost = cost.clone()
    batch_shape = cost.shape[:-2]
    N, K = cost.shape[-2], cost.shape[-1]
    points = min(N, K)
    num_region = int(torch.LongTensor(list(batch_shape)).prod())
    cost = cost.view(num_region, N, K)
    assignment = - np.ones((num_region, points))
    for k in range(num_region):
        _, assignment_k = linear_sum_assignment(cost[k].permute(1, 0).detach().cpu().numpy(), maximize=False)
        assignment[k] = assignment_k
    assignment = torch.LongTensor(assignment).to(cost.device)
    assignment = assignment.view(*batch_shape, points)
    return assignment
