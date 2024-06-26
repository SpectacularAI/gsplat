"""Python bindings for 3D gaussian projection"""

from typing import Optional, Tuple

import numpy
import torch
from jaxtyping import Float
from torch import Tensor
from torch.autograd import Function

import gsplat.cuda as _C
from gsplat._torch_impl import project_gaussians_forward as torch_project_gaussians

def project_gaussians(
    means3d: Float[Tensor, "*batch 3"],
    scales: Float[Tensor, "*batch 3"],
    glob_scale: float,
    quats: Float[Tensor, "*batch 4"],
    linear_velocity: Optional[Float[Tensor, "3"]],
    angular_velocity: Optional[Float[Tensor, "3"]],
    rolling_shutter_time: float,
    exposure_time: float,
    viewmat: Float[Tensor, "4 4"],
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    img_height: int,
    img_width: int,
    block_width: int,
    clip_thresh: float = 0.01,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """This function projects 3D gaussians to 2D using the EWA splatting method for gaussian splatting.

    Note:
        This function is differentiable w.r.t the means3d, scales and quats inputs.

    Args:
       means3d (Tensor): xyzs of gaussians.
       scales (Tensor): scales of the gaussians.
       glob_scale (float): A global scaling factor applied to the scene.
       quats (Tensor): rotations in normalized quaternion [w,x,y,z] format.
       linear_velocity (Tuple): Camera linear velocity in camera coordinates (scene units / s)
       angular_velocity (Tuple): Camera angular velocity in camera coordinates (scene units / s)
       rolling_shutter_time (float): rollings shutter roll time, seconds
       viewmat (Tensor): view matrix for rendering.
       fx (float): focal length x.
       fy (float): focal length y.
       cx (float): principal point x.
       cy (float): principal point y.
       img_height (int): height of the rendered image.
       img_width (int): width of the rendered image.
       block_width (int): side length of tiles inside projection/rasterization in pixels (always square). 16 is a good default value, must be between 2 and 16 inclusive.
       clip_thresh (float): minimum z depth threshold.

    Returns:
        A tuple of {Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor}:

        - **xys** (Tensor): x,y locations of 2D gaussian projections.
        - **depths** (Tensor): z depth of gaussians.
        - **pix_vels** (Tensor): pixel velocities
        - **radii** (Tensor): radii of 2D gaussian projections.
        - **conics** (Tensor): conic parameters for 2D gaussian.
        - **compensation** (Tensor): the density compensation for blurring 2D kernel
        - **num_tiles_hit** (Tensor): number of tiles hit per gaussian.
        - **cov3d** (Tensor): 3D covariances.
    """
    assert block_width > 1 and block_width <= 16, "block_width must be between 2 and 16"
    assert (quats.norm(dim=-1) - 1 < 1e-6).all(), "quats must be normalized"

    if linear_velocity is None:
        assert angular_velocity is None
        assert rolling_shutter_time == 0
        v_lin = torch.tensor([0, 0, 0], dtype=means3d.dtype)
        v_ang = v_lin
        use_torch = False
    else:
        assert angular_velocity is not None
        v_lin = linear_velocity
        v_ang = angular_velocity
        use_torch = linear_velocity.requires_grad or angular_velocity.requires_grad

    if use_torch:
        viewmat4x4 = torch.vstack([viewmat, torch.Tensor([[0, 0, 0, 1]]).to(viewmat)]).contiguous()
        (
            cov3d,
            _cov2d,
            xys,
            depths,
            pix_vels,
            radii,
            conics,
            compensation,
            num_tiles_hit,
            _masks,
        ) = torch_project_gaussians(
            means3d.contiguous(),
            scales.contiguous(),
            glob_scale,
            quats.contiguous(),
            torch.ravel(v_lin),
            torch.ravel(v_ang),
            rolling_shutter_time,
            exposure_time,
            viewmat4x4,
            (fx, fy, cx, cy),
            (img_width, img_height),
            block_width,
            clip_thresh,
        )

        return (xys, depths, pix_vels, radii, conics, compensation, num_tiles_hit, cov3d)

    return _ProjectGaussians.apply(
        means3d.contiguous(),
        scales.contiguous(),
        glob_scale,
        quats.contiguous(),
        v_lin,
        v_ang,
        rolling_shutter_time,
        exposure_time,
        viewmat.contiguous(),
        fx,
        fy,
        cx,
        cy,
        img_height,
        img_width,
        block_width,
        clip_thresh,
    )


class _ProjectGaussians(Function):
    """Project 3D gaussians to 2D."""

    @staticmethod
    def forward(
        ctx,
        means3d: Float[Tensor, "*batch 3"],
        scales: Float[Tensor, "*batch 3"],
        glob_scale: float,
        quats: Float[Tensor, "*batch 4"],
        linear_velocity: Float[Tensor, "3"],
        angular_velocity: Float[Tensor, "3"],
        rolling_shutter_time: float,
        exposure_time: float,
        viewmat: Float[Tensor, "4 4"],
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        img_height: int,
        img_width: int,
        block_width: int,
        clip_thresh: float = 0.01,
    ):
        num_points = means3d.shape[-2]
        if num_points < 1 or means3d.shape[-1] != 3:
            raise ValueError(f"Invalid shape for means3d: {means3d.shape}")

        (
            cov3d,
            xys,
            depths,
            pix_vels,
            radii,
            conics,
            compensation,
            num_tiles_hit,
        ) = _C.project_gaussians_forward(
            num_points,
            means3d,
            scales,
            glob_scale,
            quats,
            tuple(numpy.ravel(linear_velocity.detach().tolist())),
            tuple(numpy.ravel(angular_velocity.detach().tolist())),
            rolling_shutter_time,
            exposure_time,
            viewmat,
            fx,
            fy,
            cx,
            cy,
            img_height,
            img_width,
            block_width,
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
        ctx.linear_velocity = linear_velocity
        ctx.angular_velocity = angular_velocity
        ctx.rolling_shutter_time = rolling_shutter_time
        ctx.exposure_time = exposure_time

        # Save tensors.
        ctx.save_for_backward(
            means3d,
            scales,
            quats,
            viewmat,
            cov3d,
            radii,
            conics,
            compensation,
        )

        return (xys, depths, pix_vels, radii, conics, compensation, num_tiles_hit, cov3d)

    @staticmethod
    def backward(
        ctx,
        v_xys,
        v_depths,
        v_pix_vels,
        v_radii,
        v_conics,
        v_compensation,
        v_num_tiles_hit,
        v_cov3d,
    ):
        (
            means3d,
            scales,
            quats,
            viewmat,
            cov3d,
            radii,
            conics,
            compensation,
        ) = ctx.saved_tensors

        (v_cov2d, v_cov3d, v_mean3d, v_scale, v_quat) = _C.project_gaussians_backward(
            ctx.num_points,
            means3d,
            scales,
            ctx.glob_scale,
            quats,
            tuple(numpy.ravel(ctx.linear_velocity.tolist())),
            tuple(numpy.ravel(ctx.angular_velocity.tolist())),
            ctx.rolling_shutter_time,
            ctx.exposure_time,
            viewmat,
            ctx.fx,
            ctx.fy,
            ctx.cx,
            ctx.cy,
            ctx.img_height,
            ctx.img_width,
            cov3d,
            radii,
            conics,
            compensation,
            v_xys,
            v_depths,
            v_pix_vels,
            v_conics,
            v_compensation,
        )

        if viewmat.requires_grad:
            v_viewmat = torch.zeros_like(viewmat)
            R = viewmat[..., :3, :3]

            # Denote ProjectGaussians for a single Gaussian (mean3d, q, s)
            # viemwat = [R, t] as:
            #
            #   f(mean3d, q, s, R, t, intrinsics)
            #       = g(R @ mean3d + t,
            #           R @ cov3d_world(q, s) @ R^T ))
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
            # Gradients for R and t can then be obtained by summing over
            # all the Gaussians.
            v_mean3d_cam = torch.matmul(v_mean3d, R.transpose(-1, -2))

            # gradient w.r.t. view matrix translation
            v_viewmat[..., :3, 3] = v_mean3d_cam.sum(-2)

            # gradent w.r.t. view matrix rotation
            for j in range(3):
                for l in range(3):
                    v_viewmat[..., j, l] = torch.dot(
                        v_mean3d_cam[..., j], means3d[..., l]
                    )
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
            # linear_velocity: Optional[Float[Tensor, "3"]],
            None,
            # angular_velocity: Optional[Float[Tensor, "3"]],
            None,
            # rolling_shutter_time: float,
            None,
            # exposure_time: float,
            None,
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
            # block_width: int,
            None,
            # clip_thresh,
            None,
        )
