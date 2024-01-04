"""Python bindings for 3D gaussian projection"""

from typing import Tuple

import math
import torch
from jaxtyping import Float
from torch import Tensor
from torch.autograd import Function

import gsplat.cuda as _C

def projection_matrix(znear, zfar, fovx, fovy, device):
    """
    Constructs an OpenGL-style perspective projection matrix.
    """
    t = znear * math.tan(0.5 * fovy)
    b = -t
    r = znear * math.tan(0.5 * fovx)
    l = -r
    n = znear
    f = zfar
    return torch.tensor(
        [
            [2 * n / (r - l), 0.0, (r + l) / (r - l), 0.0],
            [0.0, 2 * n / (t - b), (t + b) / (t - b), 0.0],
            [0.0, 0.0, (f + n) / (f - n), -1.0 * f * n / (f - n)],
            [0.0, 0.0, 1.0, 0.0],
        ],
        device=device,
    )

class ProjectGaussians(Function):
    """This function projects 3D gaussians to 2D using the EWA splatting method for gaussian splatting.

    Args:
       means3d (Tensor): xyzs of gaussians.
       scales (Tensor): scales of the gaussians.
       glob_scale (float): A global scaling factor applied to the scene.
       quats (Tensor): rotations in quaternion [w,x,y,z] format.
       viewmat (Tensor): view matrix for rendering.
       projmat (Tensor): projection matrix for rendering.
       fx (float): focal length x.
       fy (float): focal length y.
       img_height (int): height of the rendered image.
       img_width (int): width of the rendered image.
       tile_bounds (Tuple): tile dimensions as a len 3 tuple (tiles.x , tiles.y, 1).
       clip_thresh (float): minimum z depth threshold.

    Returns:
        A tuple of {Tensor, Tensor, Tensor, Tensor, int, Tensor}:

        - **xys** (Tensor): x,y locations of 2D gaussian projections.
        - **depths** (Tensor): z depth of gaussians.
        - **radii** (Tensor): radii of 2D gaussian projections.
        - **conics** (Tensor): conic parameters for 2D gaussian.
        - **num_tiles_hit** (int): number of tiles hit.
        - **cov3d** (Tensor): 3D covariances.
    """

    @staticmethod
    def forward(
        ctx,
        means3d: Float[Tensor, "*batch 3"],
        scales: Float[Tensor, "*batch 3"],
        glob_scale: float,
        quats: Float[Tensor, "*batch 4"],
        viewmat: Float[Tensor, "4 4"],
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        img_height: int,
        img_width: int,
        tile_bounds: Tuple[int, int, int],
        clip_thresh: float = 0.01,
    ):
        num_points = means3d.shape[-2]

        Z_NEAR = 0.001
        Z_FAR = 1000
        fovx = 2 * math.atan(img_width / (2 * fx))
        fovy = 2 * math.atan(img_height / (2 * fy))
        projmat = projection_matrix(Z_NEAR, Z_FAR, fovx, fovy, device=viewmat.device)

        (
            cov3d,
            xys,
            depths,
            radii,
            conics,
            num_tiles_hit,
        ) = _C.project_gaussians_forward(
            num_points,
            means3d,
            scales,
            glob_scale,
            quats,
            viewmat,
            projmat,
            fx,
            fy,
            cx,
            cy,
            img_height,
            img_width,
            tile_bounds,
            clip_thresh,
        )

        # Save non-tensors.
        ctx.img_height = img_height
        ctx.img_width = img_width
        ctx.num_points = num_points
        ctx.glob_scale = glob_scale
        ctx.fx = fx
        ctx.fy = fy
        ctx.cx = cx
        ctx.cy = cy

        # Save tensors.
        ctx.save_for_backward(
            means3d,
            scales,
            quats,
            viewmat,
            projmat,
            cov3d,
            radii,
            conics,
        )

        return (xys, depths, radii, conics, num_tiles_hit, cov3d)

    @staticmethod
    def backward(ctx, v_xys, v_depths, v_radii, v_conics, v_num_tiles_hit, v_cov3d):
        (
            means3d,
            scales,
            quats,
            viewmat,
            projmat,
            cov3d,
            radii,
            conics,
        ) = ctx.saved_tensors

        (v_cov2d, v_cov3d, v_mean3d, v_scale, v_quat) = _C.project_gaussians_backward(
            ctx.num_points,
            means3d,
            scales,
            ctx.glob_scale,
            quats,
            viewmat,
            projmat,
            ctx.fx,
            ctx.fy,
            ctx.cx,
            ctx.cy,
            ctx.img_height,
            ctx.img_width,
            cov3d,
            radii,
            conics,
            v_xys,
            v_depths,
            v_conics,
        )

        if viewmat.requires_grad:
            v_viewmat = torch.zeros_like(viewmat)
            R = viewmat[..., :3, :3]

            # Approximate gradient for f(.) = loss(F(ProjectGaussians(.))).
            # Denote:
            #
            #   f(viewmat, means3d, ...) =
            #       g(p_view(viewmat, means3d),
            #         cov2d(viewmat, means3d, ...), ...)
            #
            # The derivative w.r.t. the element (i,j) of the view matrix
            # (viewmat[i, j] := vij) is
            #
            #   v_viewmat[i, j] := d f / d vij = \sum_k (
            #       \sum_l d g / d p_view[k, l] * d p_view[k, l] / d vij +
            #       \sum_lm d g / d cov2d[k, l, m] * d cov2d[k, l, m] / d vij
            #   )
            #
            # In the context of fine tuning camera poses, it is reasonable to
            # assume that the projected shapes of the Gaussians (cov2d) are
            # rather insensitive to small perturbations of the poses, when
            # compared to the means of Gaussians in camera coordinates:
            #
            #   p_view(viewmat, means3d)[k, j]
            #     = (viewmat[:3, :3] @ means3d[k, :]^T + viewmat[:3, 3])^T[j]
            #     =: (R @ means3d[k, :]^T + t)^T[j]
            #     = (means3d[k, :] @ R^T + t^T)[j]
            #     = \sum_l means3d[k, l] * R[j, l] + t[j]
            #
            # That is:
            #
            #   d f / d vij =: v_viewmat[i, j]
            #       =~ \sum_k dot(d g / d p_view[k, :], d p_view[k, :] / d vij)
            #       =: \sum_k dot(v_mean3d_cam[k, :], d p_view[k, :] / d vij)
            #
            # and
            #
            #   v_mean3d[k, i] := d f / d mean3d[k, i]
            #       =~ \sum_j d g / d p_view[k, j] * d p_view[k, j] / d mean3d[k, i]
            #       = \sum_j d g / d p_view[k, j] * R[j, i]
            #       = dot(d g / d p_view[k, :] * R[:, i])
            #       = (v_mean3d_cam[k, :] @ R)[i]
            #
            #   => v_mean3d_cam[k, :] =~ v_mean3d[k, :] @ R^T
            #
            v_mean3d_cam = torch.matmul(v_mean3d, R.transpose(-1, -2))

            # gradient w.r.t. view matrix translation
            #
            #   d p_view[k, i] / d t[j] = delta_ij
            #   => d p_view[k, :] / dt = I
            #   => d f / d t = \sum_k v_mean_cam[k, :]
            #
            v_viewmat[..., :3, 3] = v_mean3d_cam.sum(-2)

            # gradent w.r.t. view matrix rotation
            #
            #   d p_view[k, i] / d R[j, l] = means3d[k, l] * delta_ij
            #
            # => d f / d v R[j, l]
            #   = \sum_k dot(v_mean3d_cam[k, :], d p_view[k, :] / d R[j, l])
            #   = \sum_ik v_mean3d_cam[k, i] * means3d[k, l] * delta_ij
            #   = \sum_k v_mean3d_cam[k, j] * means3d[k, l]
            #
            for j in range(3):
                for l in range(3):
                    v_viewmat[..., j, l] = torch.dot(v_mean3d_cam[..., j], means3d[..., l])
        else:
            v_viewmat = None

        # Return a gradient for each input.
        return (
            # means3d: Float[Tensor, "*batch 3"],
            v_mean3d,
            # scales: Float[Tensor, "*batch 3"],
            v_scale,
            # glob_scale: float,
            None,
            # quats: Float[Tensor, "*batch 4"],
            v_quat,
            # viewmat: Float[Tensor, "4 4"],
            v_viewmat,
            # fx: float,
            None,
            # fy: float,
            None,
            # cx: float,
            None,
            # cy: float,
            None,
            # img_height: int,
            None,
            # img_width: int,
            None,
            # tile_bounds: Tuple[int, int, int],
            None,
        )
