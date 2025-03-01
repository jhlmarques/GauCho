# Copyright (c) OpenMMLab. All rights reserved.
import math
import mmcv
import numpy as np
from mmrotate.core import bbox
import torch
from mmdet.core.bbox.coder.base_bbox_coder import BaseBBoxCoder
from mmrotate.core.bbox.transforms import hbb2obb

from ..builder import ROTATED_BBOX_CODERS
from ..transforms import norm_angle


@ROTATED_BBOX_CODERS.register_module()
class GauchoAnchorOBBDecoder(BaseBBoxCoder):
    """
    Gaussian-Cholesky Oriented Bounding Box coder. 
    Decodes gaussian anchor deltas (dx, dy, da, db, dc) into (mux, muy, a, b, c) gaussian bounding boxes
    and optionally further into (x, y, w, h, theta) oriented bounding boxes

    """

    def __init__(self,
                 target_means=(0., 0., 0., 0., 0.),
                 target_stds=(1., 1., 1., 1., 1.),
                 angle_range='le90',
                 norm_factor=None,
                 edge_swap=False,
                 proj_xy=False,
                 add_ctr_clamp=False,
                 ctr_clamp=32,
                 horizontal=False):
        super(BaseBBoxCoder, self).__init__()
        self.means = target_means
        self.stds = target_stds
        self.add_ctr_clamp = add_ctr_clamp
        self.ctr_clamp = ctr_clamp
        self.angle_range = angle_range
        self.norm_factor = norm_factor
        self.edge_swap = edge_swap
        self.proj_xy = proj_xy
        self.horizontal = horizontal

    def encode(self, bboxes, gt_bboxes):
        """Get box regression transformation deltas that can be used to
        transform the ``bboxes`` into the ``gt_bboxes``.

        Does not properly compute deltas for cholesky elements. This is intended as a placeholder fix for KFIoU integration,
        specifically for the x and y deltas

        Args:
            bboxes (torch.Tensor): Source boxes, e.g., object proposals.
            gt_bboxes (torch.Tensor): Target of the transformation, e.g.,
                ground-truth boxes.

        Returns:
            torch.Tensor: Box transformation deltas
        """
        assert bboxes.size(0) == gt_bboxes.size(0)
        
        if bboxes.size(-1) == 4:
            bboxes = hbb2obb(bboxes, self.angle_range)
        
        assert bboxes.size(-1) == 5
        assert gt_bboxes.size(-1) == 5
        if self.angle_range in ['oc', 'le135', 'le90']:
            return bbox2delta(bboxes, gt_bboxes, self.means, self.stds,
                              self.angle_range, self.norm_factor,
                              self.edge_swap, self.proj_xy)
        else:
            raise NotImplementedError

    def decode(self,
               bboxes,
               pred_bboxes,
               max_shape=None,
               wh_ratio_clip=16 / 1000,
               to_obb=True):
        """Apply transformation `pred_bboxes` to `boxes`.

        Args:
            bboxes (torch.Tensor): Basic boxes. Shape (B, N, 5) or (N, 5)
            pred_bboxes (torch.Tensor): Encoded offsets with respect to each \
                roi. Has shape (B, N, num_classes * 5) or (B, N, 5) or \
               (N, num_classes * 5) or (N, 5). Note N = num_anchors * W * H \
               when rois is a grid of anchors.
            max_shape (Sequence[int] or torch.Tensor or Sequence[ \
               Sequence[int]],optional): Maximum bounds for boxes, specifies \
               (H, W, C) or (H, W). If bboxes shape is (B, N, 5), then \
               the max_shape should be a Sequence[Sequence[int]] \
               and the length of max_shape should also be B.
            wh_ratio_clip (float, optional): The allowed ratio between
                width and height.
            to_obb (bool, optional): If True, further decode gaussian bounding boxes 
                into oriented bounding boxes. Defaults to True.

        Returns:
            torch.Tensor: Decoded boxes.
        """
        assert pred_bboxes.size(0) == bboxes.size(0)
        
        if bboxes.size(-1) == 4:
            bboxes = hbb2obb(bboxes, self.angle_range)

        if self.angle_range not in ['oc', 'le135', 'le90']:
            raise NotImplementedError
        else:
            return delta2bbox(bboxes, pred_bboxes, self.means, self.stds,
                              max_shape, wh_ratio_clip, self.add_ctr_clamp,
                              self.ctr_clamp, self.angle_range,
                              self.norm_factor, self.edge_swap, self.proj_xy, to_obb, self.horizontal)

@mmcv.jit(coderize=True)
def bbox2delta(proposals,
               gt,
               means=(0., 0., 0., 0., 0.),
               stds=(1., 1., 1., 1., 1.),
               angle_range='oc',
               norm_factor=None,
               edge_swap=False,
               proj_xy=False):
    """
    Does not properly compute deltas for cholesky elements. This is intended as a placeholder fix for KFIoU integration,
    specifically for the x and y deltas

    Args:
        proposals (torch.Tensor): Boxes to be transformed, shape (N, ..., 5)
        gt (torch.Tensor): Gt bboxes to be used as base, shape (N, ..., 5)
        means (Sequence[float]): Denormalizing means for delta coordinates
        stds (Sequence[float]): Denormalizing standard deviation for delta
            coordinates.
        angle_range (str, optional): Angle representations. Defaults to 'oc'.
        norm_factor (None|float, optional): Regularization factor of angle.
        edge_swap (bool, optional): Whether swap the edge if w < h.
            Defaults to False.
        proj_xy (bool, optional): Whether project x and y according to angle.
            Defaults to False.

    Returns:
        Tensor: deltas with shape (N, 5), where columns represent dx, dy,
            dw, dh, da.
    """
    assert proposals.size() == gt.size()
    proposals = proposals.float()
    gt = gt.float()
    px, py, pw, ph, pa = proposals.unbind(dim=-1)
    gx, gy, gw, gh, ga = gt.unbind(dim=-1)

    if proj_xy:
        dx = (torch.cos(pa) * (gx - px) + torch.sin(pa) * (gy - py)) / pw
        dy = (-torch.sin(pa) * (gx - px) + torch.cos(pa) * (gy - py)) / ph
    else:
        dx = (gx - px) / pw
        dy = (gy - py) / ph

    dw = pw
    dh = ph
    da = pa

    # if edge_swap:
    #     dtheta1 = norm_angle(ga - pa, angle_range)
    #     dtheta2 = norm_angle(ga - pa + np.pi / 2, angle_range)
    #     abs_dtheta1 = torch.abs(dtheta1)
    #     abs_dtheta2 = torch.abs(dtheta2)
    #     gw_regular = torch.where(abs_dtheta1 < abs_dtheta2, gw, gh)
    #     gh_regular = torch.where(abs_dtheta1 < abs_dtheta2, gh, gw)
    #     da = torch.where(abs_dtheta1 < abs_dtheta2, dtheta1, dtheta2)
    #     dw = torch.log(gw_regular / pw)
    #     dh = torch.log(gh_regular / ph)
    # else:
    #     da = norm_angle(ga - pa, angle_range)
    #     dw = torch.log(gw / pw)
    #     dh = torch.log(gh / ph)

    # if norm_factor:
    #     da /= norm_factor * np.pi

    deltas = torch.stack([dx, dy, dw, dh, da], dim=-1)
    means = deltas.new_tensor(means).unsqueeze(0)
    stds = deltas.new_tensor(stds).unsqueeze(0)
    deltas = deltas.sub_(means).div_(stds)
    return deltas

@mmcv.jit(coderize=True)
def delta2bbox(rois,
               deltas,
               means=(0., 0., 0., 0., 0.),
               stds=(1., 1., 1., 1., 1.),
               max_shape=None,
               wh_ratio_clip=16 / 1000,
               add_ctr_clamp=False,
               ctr_clamp=32,
               angle_range='oc',
               norm_factor=None,
               edge_swap=False,
               proj_xy=False,
               to_obb=True,
               horizontal=False,):
    """Apply deltas to shift/scale base boxes. Typically the rois are anchor
    or proposed bounding boxes and the deltas are network outputs used to
    shift/scale those boxes. This is the inverse function of
    :func:`bbox2delta`.

    Args:
        rois (torch.Tensor): Boxes to be transformed. Has shape (N, 5).
        deltas (torch.Tensor): Encoded offsets relative to each roi.
            Has shape (N, num_classes * 5) or (N, 5). Note
            N = num_base_anchors * W * H, when rois is a grid of
            anchors.
        means (Sequence[float]): Denormalizing means for delta coordinates.
            Default (0., 0., 0., 0., 0.).
        stds (Sequence[float]): Denormalizing standard deviation for delta
            coordinates. Default (1., 1., 1., 1., 1.).
        max_shape (Sequence[int] or torch.Tensor or Sequence[
            Sequence[int]],optional): Maximum bounds for boxes, specifies
           (H, W, C) or (H, W). If bboxes shape is (B, N, 5), then
           the max_shape should be a Sequence[Sequence[int]]
           and the length of max_shape should also be B.
        wh_ratio_clip (float): Maximum aspect ratio for boxes. Default
            16 / 1000.
        add_ctr_clamp (bool): Whether to add center clamp, when added, the
            predicted box is clamped is its center is too far away from
            the original anchor's center. Only used by YOLOF. Default False.
        ctr_clamp (int): the maximum pixel shift to clamp. Only used by
            YOLOF. Default 32.
        angle_range (str, optional): Angle representations. Defaults to 'oc'.
        norm_factor (None|float, optional): Regularization factor of angle.
        edge_swap (bool, optional): Whether swap the edge if w < h.
            Defaults to False.
        proj_xy (bool, optional): Whether project x and y according to angle.
            Defaults to False.
        to_obb (bool, optional): If True, further decode gaussian bounding boxes 
            into oriented bounding boxes. Defaults to True.
        wh_scale_factor (float): Scalar value of width and height in OBB -> Gaussian conversion
        min_scaled_ar (float): Minimum aspect ratio for reescaled anchors (gamma regression)

    Returns:
        Tensor: Boxes with shape (N, num_classes * 5) or (N, 5), where 5
           represent cx, cy, w, h, a.
    """
    means = deltas.new_tensor(means).view(1, -1).repeat(1, deltas.size(1) // 5)
    stds = deltas.new_tensor(stds).view(1, -1).repeat(1, deltas.size(1) // 5)
    denorm_deltas = deltas * stds + means

    dx = denorm_deltas[:, 0::5]
    dy = denorm_deltas[:, 1::5]
    dalpha = denorm_deltas[:, 2::5]
    dbeta = denorm_deltas[:, 3::5]
    dgamma = denorm_deltas[:, 4::5]

    # Compute center of each roi
    px = rois[:, 0].unsqueeze(1).expand_as(dx)
    py = rois[:, 1].unsqueeze(1).expand_as(dy)
    # Compute width/height of each roi
    pw = rois[:, 2].unsqueeze(1).expand_as(dalpha)
    ph = rois[:, 3].unsqueeze(1).expand_as(dbeta)
    # Compute rotated angle of each roi
    pa = rois[:, 4].unsqueeze(1).expand_as(dgamma)
    dx_width = pw * dx
    dy_height = ph * dy
    max_ratio = np.abs(np.log(wh_ratio_clip))
    
    # if add_ctr_clamp:
    #     raise NotImplemented
    # else:
    #     dalpha = dalpha.clamp(min=-max_ratio, max=max_ratio)
    #     dbeta = dbeta.clamp(min=-max_ratio, max=max_ratio)

    # Due to the nature of OBB->GBB conversion, the highest eigenvalue
    # should be the width and the lowest the height
    eig_w = 0.25 * pw.square()
    eig_h = 0.25 * ph.square()
    sqrt_eig_w = 0.5 * pw
    sqrt_eig_h = 0.5 * ph

    # This assumes that horizontal anchors can have w < h (oc encoding)
    if horizontal:
        galpha      = sqrt_eig_w * dalpha.exp()
        gbeta       = sqrt_eig_h * dbeta.exp()
        ggamma      = torch.max(torch.min(sqrt_eig_w, sqrt_eig_h), torch.abs(sqrt_eig_w - sqrt_eig_h)) * dgamma
    # Input anchor is oriented
    else:
        p_cova  = torch.cos(pa).square() * eig_w + torch.sin(pa).square() * eig_h
        #p_covb  = torch.cos(pa).square() * eig_h + torch.sin(pa).square() * eig_w
        p_covc  = 0.5 * torch.sin(2*pa) * (eig_w - eig_h)
        palpha  = torch.sqrt(p_cova)
        pgamma  = p_covc / palpha
        pbeta  = (sqrt_eig_h * sqrt_eig_w).clamp(1.0) / palpha
        #pbeta   = torch.sqrt((p_covb + 5e-2) - (p_covc.square() / p_cova))

        alpha_scale = dalpha.exp()
        beta_scale  = dbeta.exp()
        gamma_add   = sqrt_eig_h.clamp(1.0) * dgamma
        galpha      = palpha * alpha_scale
        gbeta       = pbeta  * beta_scale
        ggamma      = pgamma + gamma_add

    # Use network energy to shift the center of each roi
    if proj_xy:
        gx = dx * pw * torch.cos(pa) - dy * ph * torch.sin(pa) + px
        gy = dx * pw * torch.sin(pa) + dy * ph * torch.cos(pa) + py
    else:
        gx = px + dx_width
        gy = py + dy_height

    if max_shape is not None:
        gx = gx.clamp(min=0, max=max_shape[1] - 1)
        gy = gy.clamp(min=0, max=max_shape[0] - 1)

    # Convert covariance values into box width, height and angle
    ga = galpha **2
    gb = gbeta **2 + ggamma **2
    gc = galpha * ggamma

    if to_obb:
        delta = torch.sqrt(4 * torch.abs(gc).square() + (ga - gb).square())
        eig1 = 0.5 * (ga + gb - delta)
        eig2 = 0.5 * (ga + gb + delta)
        gw = 2 * torch.sqrt(eig2)
        gh = 2 * torch.sqrt(eig1)
        gt = torch.atan(gc / (eig2 - gb))
        gt = norm_angle(gt, angle_range)

        decoded_bbox = torch.stack([gx, gy, gw, gh, gt],dim=-1).view(deltas.size())
    else:
        decoded_bbox = torch.stack([gx, gy, ga, gb, gc],dim=-1).view(deltas.size())

    return decoded_bbox