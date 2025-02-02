from einops import repeat
import torch
from torch import nn
import torch.nn.functional as F

from pytorch3d.loss.chamfer import _handle_pointcloud_input
from pytorch3d.ops.knn import knn_gather, knn_points


def chamfer_distance_no_reduction(
    x,
    y,
    x_lengths=None,
    y_lengths=None,
    x_normals=None,
    y_normals=None,
    weights=None,
):
    """
    Chamfer distance between two pointclouds x and y.

    Args:
        x: FloatTensor of shape (N, P1, D) or a Pointclouds object representing
            a batch of point clouds with at most P1 points in each batch element,
            batch size N and feature dimension D.
        y: FloatTensor of shape (N, P2, D) or a Pointclouds object representing
            a batch of point clouds with at most P2 points in each batch element,
            batch size N and feature dimension D.
        x_lengths: Optional LongTensor of shape (N,) giving the number of points in each
            cloud in x.
        y_lengths: Optional LongTensor of shape (N,) giving the number of points in each
            cloud in y.
        x_normals: Optional FloatTensor of shape (N, P1, D).
        y_normals: Optional FloatTensor of shape (N, P2, D).
        weights: Optional FloatTensor of shape (N,) giving weights for
            batch elements for reduction operation.

    Returns:
        2-element tuple containing

        - **loss**: Tensor giving the reduced distance between the pointclouds
          in x and the pointclouds in y.
        - **loss_normals**: Tensor giving the reduced cosine distance of normals
          between pointclouds in x and pointclouds in y. Returns None if
          x_normals and y_normals are None.
    """
    # _validate_chamfer_reduction_inputs(batch_reduction, point_reduction)

    x, x_lengths, x_normals = _handle_pointcloud_input(x, x_lengths, x_normals)
    y, y_lengths, y_normals = _handle_pointcloud_input(y, y_lengths, y_normals)

    return_normals = x_normals is not None and y_normals is not None

    N, P1, D = x.shape
    P2 = y.shape[1]

    # Check if inputs are heterogeneous and create a lengths mask.
    is_x_heterogeneous = (x_lengths != P1).any()
    is_y_heterogeneous = (y_lengths != P2).any()
    x_mask = (
        torch.arange(P1, device=x.device)[None] >= x_lengths[:, None]
    )  # shape [N, P1]
    y_mask = (
        torch.arange(P2, device=y.device)[None] >= y_lengths[:, None]
    )  # shape [N, P2]

    if y.shape[0] != N or y.shape[2] != D:
        raise ValueError("y does not have the correct shape.")
    # if weights is not None:
    #     if weights.size(0) != N:
    #         raise ValueError("weights must be of shape (N,).")
    #     if not (weights >= 0).all():
    #         raise ValueError("weights cannot be negative.")
    #     if weights.sum() == 0.0:
    #         weights = weights.view(N, 1)
    #         if batch_reduction in ["mean", "sum"]:
    #             return (
    #                 (x.sum((1, 2)) * weights).sum() * 0.0,
    #                 (x.sum((1, 2)) * weights).sum() * 0.0,
    #             )
    #         return ((x.sum((1, 2)) * weights) * 0.0, (x.sum((1, 2)) * weights) * 0.0)

    cham_norm_x = x.new_zeros(())
    cham_norm_y = x.new_zeros(())

    x_nn = knn_points(x, y, lengths1=x_lengths, lengths2=y_lengths, K=1)
    y_nn = knn_points(y, x, lengths1=y_lengths, lengths2=x_lengths, K=1)

    cham_x = x_nn.dists[..., 0]  # (N, P1)
    cham_y = y_nn.dists[..., 0]  # (N, P2)

    if is_x_heterogeneous:
        cham_x[x_mask] = 0.0
    if is_y_heterogeneous:
        cham_y[y_mask] = 0.0

    if weights is not None:
        cham_x *= weights.view(N, 1)
        cham_y *= weights.view(N, 1)

    if return_normals:
        # Gather the normals using the indices and keep only value for k=0
        x_normals_near = knn_gather(y_normals, x_nn.idx, y_lengths)[..., 0, :]
        y_normals_near = knn_gather(x_normals, y_nn.idx, x_lengths)[..., 0, :]

        cham_norm_x = 1 - torch.abs(
            F.cosine_similarity(x_normals, x_normals_near, dim=2, eps=1e-6)
        )
        cham_norm_y = 1 - torch.abs(
            F.cosine_similarity(y_normals, y_normals_near, dim=2, eps=1e-6)
        )

        if is_x_heterogeneous:
            cham_norm_x[x_mask] = 0.0
        if is_y_heterogeneous:
            cham_norm_y[y_mask] = 0.0

        if weights is not None:
            cham_norm_x *= weights.view(N, 1)
            cham_norm_y *= weights.view(N, 1)
    
    return (cham_x, cham_y, cham_norm_x, cham_norm_y)


def pruned_chamfer_loss(x, y, 
                        x_normals, y_normals,
                        dist_thr=None, normals_thr=None, 
                        mask_x=None, mask_y=None, 
                        device='cuda:0'):
    
    if mask_x is None:
        x_masked = x
        x_normals_masked = x_normals.unsqueeze(0)
    else:
        # print('masking')
        x_masked = x[mask_x]
        if x_normals is not None:
            x_normals_masked = x_normals[mask_x].unsqueeze(0)
    
    if mask_y is None:
        y_masked = y
        y_normals_masked = y_normals.unsqueeze(0)
    else:
        # print('masking')
        y_masked = y[mask_y]
        if y_normals is not None:
            y_normals_masked = y_normals[mask_y].unsqueeze(0)
    
    cham_x, cham_y, cham_norm_x, cham_norm_y = chamfer_distance_no_reduction(
        x_masked.unsqueeze(0),
        y_masked.unsqueeze(0),
        x_normals=x_normals_masked,
        y_normals=y_normals_masked,
    )
    cham_x, cham_y, cham_norm_x, cham_norm_y = cham_x[0], cham_y[0], cham_norm_x[0], cham_norm_y[0]
    
    loss_val = torch.zeros(1).to(device)
    
    cham_x_mask = torch.ones_like(cham_x).bool()
    cham_y_mask = torch.ones_like(cham_y).bool()
    if dist_thr is not None:
        cham_x_mask &= (cham_x < dist_thr)
        cham_y_mask &= (cham_y < dist_thr)
    if normals_thr is not None:
        cham_x_mask &= (cham_norm_x < normals_thr)
        cham_y_mask &= (cham_norm_y < normals_thr)
    
    if cham_x_mask.sum() > 0:
        loss_val += cham_x[cham_x_mask].mean()
    if cham_y_mask.sum() > 0:
        loss_val += cham_y[cham_y_mask].mean()
    return loss_val
