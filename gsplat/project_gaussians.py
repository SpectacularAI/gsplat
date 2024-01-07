"""Python bindings for 3D gaussian projection"""

from typing import Optional, Tuple

import torch
from jaxtyping import Float
from torch import Tensor
from torch.autograd import Function

import gsplat.cuda as _C

class ProjectGaussians(Function):
    """This function projects 3D gaussians to 2D using the EWA splatting method for gaussian splatting.

    Args:
       means3d (Tensor): xyzs of gaussians.
       scales (Tensor): scales of the gaussians.
       glob_scale (float): A global scaling factor applied to the scene.
       quats (Tensor): rotations in quaternion [w,x,y,z] format.
       viewmat (Tensor): view matrix for rendering.
       projmat (Tensor): DEPRECATED and ignored. Set to None
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
        projmat: Optional[Float[Tensor, "4 4"]],
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
        ctx.cx = cx
        ctx.cy = cy

        # Save tensors.
        ctx.save_for_backward(
            means3d,
            scales,
            quats,
            viewmat,
            fx,
            fy,
            xys,
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
            fx,
            fy,
            xys,
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
            fx,
            fy,
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

        ROTATION_GRADIENT_SCALING = 1

        if viewmat.requires_grad:
            v_viewmat = torch.zeros_like(viewmat)
            R = viewmat[..., :3, :3]

            # Denote ProjectGaussians for a single Gaussian (mean3d, q, s)
            # viemwat = [R, t], and intrinsics = [fx, fy, cx, cy] as:
            #
            #   f(mean3d, q, s, R, t, intrinsics)
            #       = projectCam(intrinsics,
            #           projectNormalized(
            #               R @ mean3d + t,
            #               R @ cov3d_world(q, s) @ R^T )))
            #
            # Then, the Jacobian w.r.t., t is:
            #
            #   d f / d t = df / d mean3d @ R^T
            #
            # and, in the context of fine tuning camera poses, it is reasonable
            # to assume that
            #
            #   d f / d R_ij =~ \sum_l d f / d t_l * d (R @ mean3d)_l / d R_ij
            #                = d f / d_t_i * mean3d[j]
            #
            # Furthermore, for a pinhole camera:
            #
            #   projectCam: [p_xy_norm, cov_2d_norm, depth]
            #   -> [F * p_xy_norm + [cx; cy], F * cov_2d_norm * F^T, depth]
            #
            # where F = diag([fx, fy]).
            #
            # Gradients for R, t, and intrinsics can then be obtained
            # by summing over all the Gaussians.
            v_mean3d_cam = torch.matmul(v_mean3d, R.transpose(-1, -2))

            # gradient w.r.t. view matrix translation
            v_viewmat[..., :3, 3] = v_mean3d_cam.sum(-2)

            # gradent w.r.t. view matrix rotation
            for j in range(3):
                for l in range(3):
                    v_viewmat[..., j, l] = torch.dot(v_mean3d_cam[..., j], means3d[..., l]) * ROTATION_GRADIENT_SCALING
        else:
            v_viewmat = None

        if fx.requires_grad or fy.requires_grad:
            x_norm = (xys[..., 0] - ctx.cx) / fx
            y_norm = (xys[..., 1] - ctx.cy) / fy

            #print(fx)
            #print(fy)

            # conics_norm = F^T * conics * F
            # Note: does not take blur / regularization into account
            #conics_norm_xx = conics[..., 0] * fx**2
            #conics_norm_xy = conics[..., 1] * fx*fy
            #conics_norm_yy = conics[..., 2] * fy**2
            conics_norm_xx_per_fx2 = conics[..., 0]
            conics_norm_xy_per_fxfy = conics[..., 1]
            conics_norm_yy_per_fy2 = conics[..., 2]

            v_fx = (torch.dot(x_norm, v_xys[..., 0])
                - 2 * torch.dot(conics_norm_xx_per_fx2, v_conics[..., 0]) / fx # = / fx**3
                - torch.dot(conics_norm_xy_per_fxfy, v_conics[..., 1]) / fx) # / fy / fx**2

            v_fy = (torch.dot(y_norm, v_xys[..., 1])
                - 2 * torch.dot(conics_norm_yy_per_fy2, v_conics[..., 2]) / fy # / fy**3
                - torch.dot(conics_norm_xy_per_fxfy, v_conics[..., 1]) / fy) # / fx / fy**2
        else:
            v_fx = None
            v_fy = None

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
            # projmat (deprecated)
            None,
            # fx: float,
            v_fx,
            # fy: float,
            v_fy,
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
