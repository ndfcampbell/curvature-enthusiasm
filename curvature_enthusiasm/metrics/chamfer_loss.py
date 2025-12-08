import torch
import torch.nn as nn
import numpy as np
# from utils.registry import LOSS_REGISTRY
from scipy.spatial import cKDTree


def chamfer_dist(x, y, weight_x=None, weight_y=None, full=False, return_numpy=False):
    dist = torch.cdist(x, y)
    if weight_x is None and weight_y is None:
        if full:
            loss = dist.min(-2)[0] + dist.min(-1)[0]
        else:
            loss = dist.min(-2)[0].mean(-1) + dist.min(-1)[0].mean(-1)
    else:
        if weight_y is None:
            weight_y = weight_x
        loss = (dist.min(-2)[0] * weight_x).sum(-1) / weight_x.sum(-1) + \
               (dist.min(-1)[0] * weight_y).sum(-1) / weight_y.sum(-1)

    if return_numpy:
        result = loss.detach().cpu().numpy()
        # Clean up intermediate tensors
        del dist, loss
        if weight_x is not None or weight_y is not None:
            # Only delete if they were created/modified internally
            if weight_y is weight_x and weight_x is not None:
                pass  # weight_y is just a reference, don't double-delete
        torch.cuda.empty_cache()
        return result

    return loss


def chamfer_dist_per_element(x, y):
    dist = torch.cdist(x, y)
    x_to_y = dist.min(-2)[0]
    y_to_x = dist.min(-1)[0]
    return x_to_y, y_to_x



# @LOSS_REGISTRY.register()
# class ChamferLoss(nn.Module):
#     """
#     Cross Entropy Loss
#     Args:
#         loss_weight (float, optional): Loss weight for Chamfer Distance. Default: 1.0.
#     """
#
#     def __init__(self, loss_weight=1.0):
#         super(ChamferLoss, self).__init__()
#         assert loss_weight >= 0, f'Invalid loss weight: {loss_weight}'
#         self.loss_weight = loss_weight
#
#     def forward(self, x, y, weight_x=None, weight_y=None):
#         """
#         Args:
#             x (Tensor): of shape (N, V, C). point cloud x.
#             y (Tensor): of shape (N, V, C). point cloud y.
#             weight_x (Tensor): of shape (N, V, C). weight x.
#             weight_y (Tensor): of shape (N, V, C). weight y.
#         """
#         return self.loss_weight * chamfer_dist(x, y, weight_x, weight_y).mean()

# def compute_chamfer(y_vert, y_pred, num_eval=10000):
#     # num_t = vert_sequence.shape[0]
#     # shape_x_new = shape_x.copy()
#     # shape_x_new.vert = vert_sequence[num_t - 1, ...]
#
#     # KNN using scipy's cKDTree
#     tree = cKDTree(y_vert)
#     distances, indices = tree.query(y_pred, k=1)
#
#     chamfer_curr = np.linalg.norm(y_pred - y_vert[indices], axis=1)
#
#     # Random sampling for evaluation
#     idx_eval = np.random.choice(chamfer_curr.shape[0], size=num_eval, replace=True)
#     chamfer_curr = chamfer_curr[idx_eval]
#
#     return chamfer_curr
