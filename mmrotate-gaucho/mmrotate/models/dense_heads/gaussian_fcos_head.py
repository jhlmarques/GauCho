# Copyright (c) OpenMMLab. All rights reserved.

import torch
import torch.nn as nn
from mmcv.cnn import Scale
from mmcv.runner import force_fp32
from mmdet.core import reduce_mean

from mmrotate.core import multiclass_nms_rotated
from ..builder import ROTATED_HEADS
from .rotated_fcos_head import RotatedFCOSHead
from ...core.bbox.transforms import norm_angle

import numpy as np

INF = 1e8

def xy_wh_r_2_xy_sigma(xywhr):
    """Convert oriented bounding box to 2-D Gaussian distribution.

    Args:
        xywhr (torch.Tensor): rbboxes with shape (N, 5).

    Returns:
        xy (torch.Tensor): center point of 2-D Gaussian distribution
            with shape (N, 2).
        sigma (torch.Tensor): covariance matrix of 2-D Gaussian distribution
            with shape (N, 2, 2).
    """
    _shape = xywhr.shape
    assert _shape[-1] == 5
    xy = xywhr[..., :2]
    wh = xywhr[..., 2:4].clamp(min=1e-7, max=1e7).reshape(-1, 2)
    r = xywhr[..., 4]
    cos_r = torch.cos(r)
    sin_r = torch.sin(r)
    R = torch.stack((cos_r, -sin_r, sin_r, cos_r), dim=-1).reshape(-1, 2, 2)
    S = 0.5 * torch.diag_embed(wh)

    sigma = R.bmm(S.square()).bmm(R.permute(0, 2,
                                            1)).reshape(_shape[:-1] + (2, 2))

    return xy, sigma

def decode_gaucho_offset(points, deltas, angle_range, to_obb=False):

    ctr = deltas[..., :2] + points
    cholesky = deltas[..., 2:]
    cov_a = cholesky[..., 0] ** 2
    cov_b = (cholesky[..., 1] ** 2) + (cholesky[..., 2] ** 2)
    cov_c = cholesky[..., 0] * cholesky[..., 2]

    if to_obb:
        delta = torch.sqrt(4 * torch.abs(cov_c).square() + (cov_a - cov_b).square())
        eig1 = 0.5 * (cov_a + cov_b - delta)
        eig2 = 0.5 * (cov_a + cov_b + delta)
        gw = 2 * torch.sqrt(eig2).unsqueeze(-1)
        gh = 2 * torch.sqrt(eig1).unsqueeze(-1)
        gt = torch.atan(cov_c / (eig2 - cov_b)).unsqueeze(-1)
        gt = norm_angle(gt, angle_range)
        return torch.cat([ctr, gw, gh, gt], 1)
    else:
        return torch.cat([ctr, cov_a.unsqueeze(-1), cov_b.unsqueeze(-1), cov_c.unsqueeze(-1)], 1)


@ROTATED_HEADS.register_module()
class GaussianFCOSHead(RotatedFCOSHead):

    def __init__(self,
                 gaucho_encoding=False,
                 kfiou_loss=False,
                 gaussian_centerness=False,
                 **kwargs):

        super().__init__(**kwargs)
        self.gaucho_encoding = gaucho_encoding
        self.kfiou_loss = kfiou_loss
        self.gaussian_centerness = gaussian_centerness
        self.angle_version = self.bbox_coder.angle_version

    def forward_single(self, x, scale, stride):
        """Forward features of a single scale level.

        Args:
            x (Tensor): FPN feature maps of the specified stride.
            scale (:obj: `mmcv.cnn.Scale`): Learnable scale module to resize
                the bbox prediction.
            stride (int): The corresponding stride for feature maps, only
                used to normalize the bbox prediction when self.norm_on_bbox
                is True.
        Returns:
            tuple: scores for each class, bbox predictions, angle predictions \
                and centerness predictions of input feature maps.
        """
        cls_score, bbox_pred, cls_feat, reg_feat = super(RotatedFCOSHead, self).forward_single(x)
        if self.centerness_on_reg:
            centerness = self.conv_centerness(reg_feat)
        else:
            centerness = self.conv_centerness(cls_feat)
        # scale the bbox_pred of different level
        # float to avoid overflow when enabling FP16
        bbox_pred = scale(bbox_pred).float()
        
        if self.gaucho_encoding:
            gauss_xy = bbox_pred[:, 0:2] * stride
            gaucho_ab = bbox_pred[:, 2:4].exp() * stride # cholesky a, b > 0
            gaucho_c = self.conv_angle(reg_feat) * stride
            if self.norm_on_bbox:
                raise NotImplementedError
            bbox_pred = torch.cat([gauss_xy, gaucho_ab, gaucho_c], dim=1)
            angle_pred = torch.zeros_like(centerness)
        else:
            if self.norm_on_bbox:
                # bbox_pred needed for gradient computation has been modified
                # by F.relu(bbox_pred) when run with PyTorch 1.10. So replace
                # F.relu(bbox_pred) with bbox_pred.clamp(min=0)
                bbox_pred = bbox_pred.clamp(min=0)
                if not self.training:
                    bbox_pred *= stride
            else:
                bbox_pred = bbox_pred.exp()
            angle_pred = self.conv_angle(reg_feat)
            if self.is_scale_angle:
                angle_pred = self.scale_angle(angle_pred).float()
        return cls_score, bbox_pred, angle_pred, centerness

    @force_fp32(
        apply_to=('cls_scores', 'bbox_preds', 'angle_preds', 'centernesses'))
    def loss(self,
             cls_scores,
             bbox_preds,
             angle_preds,
             centernesses,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute loss of the head.
        Args:
            cls_scores (list[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_points * num_classes.
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_points * 4.
            angle_preds (list[Tensor]): Box angle for each scale level, \
                each is a 4D-tensor, the channel number is num_points * 1.
            centernesses (list[Tensor]): centerness for each scale level, each
                is a 4D-tensor, the channel number is num_points * 1.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert len(cls_scores) == len(bbox_preds) \
               == len(angle_preds) == len(centernesses)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=bbox_preds[0].dtype,
            device=bbox_preds[0].device)
        labels, bbox_targets, angle_targets = self.get_targets(
            all_level_points, gt_bboxes, gt_labels)

        num_imgs = cls_scores[0].size(0)
        # flatten cls_scores, bbox_preds and centerness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        if self.gaucho_encoding:
            flatten_bbox_preds = [
                bbox_pred.permute(0, 2, 3, 1).reshape(-1, 5)
                for bbox_pred in bbox_preds
            ]
        else:
            flatten_bbox_preds = [
                bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
                for bbox_pred in bbox_preds
            ]

        flatten_angle_preds = [
            angle_pred.permute(0, 2, 3, 1).reshape(-1, 1)
            for angle_pred in angle_preds
        ]
        flatten_centerness = [
            centerness.permute(0, 2, 3, 1).reshape(-1)
            for centerness in centernesses
        ]
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_angle_preds = torch.cat(flatten_angle_preds)
        flatten_centerness = torch.cat(flatten_centerness)
        flatten_labels = torch.cat(labels)
        flatten_bbox_targets = torch.cat(bbox_targets)
        flatten_angle_targets = torch.cat(angle_targets)
        # repeat points to align with bbox_preds
        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points])

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((flatten_labels >= 0)
                    & (flatten_labels < bg_class_ind)).nonzero().reshape(-1)
        num_pos = torch.tensor(
            len(pos_inds), dtype=torch.float, device=bbox_preds[0].device)
        num_pos = max(reduce_mean(num_pos), 1.0)
        loss_cls = self.loss_cls(
            flatten_cls_scores, flatten_labels, avg_factor=num_pos)

        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_angle_preds = flatten_angle_preds[pos_inds]
        pos_centerness = flatten_centerness[pos_inds]
        pos_bbox_targets = flatten_bbox_targets[pos_inds]
        pos_angle_targets = flatten_angle_targets[pos_inds]
        

        if len(pos_inds) > 0:
            pos_points = flatten_points[pos_inds]

            if self.separate_angle:
                bbox_coder = self.h_bbox_coder
            else:
                bbox_coder = self.bbox_coder
                pos_bbox_preds = torch.cat([pos_bbox_preds, pos_angle_preds],
                                        dim=-1)
                pos_bbox_targets = torch.cat(
                    [pos_bbox_targets, pos_angle_targets], dim=-1)
                    
            if self.gaucho_encoding:
                pos_decoded_bbox_preds = decode_gaucho_offset(pos_points, pos_bbox_preds, self.angle_version)
            else:
                pos_decoded_bbox_preds = bbox_coder.decode(pos_points,
                                                        pos_bbox_preds)
            
            pos_decoded_target_preds = bbox_coder.decode(pos_points, pos_bbox_targets)
            pos_centerness_targets = self.centerness_target(pos_bbox_targets)

            centerness_denorm = max(reduce_mean(pos_centerness_targets.sum().detach()), 1e-6)
            
            if self.kfiou_loss:
                pos_pred_xy_offset = pos_bbox_preds[..., :2] - pos_points
                pos_target_xy_offset = pos_decoded_target_preds[..., :2] - pos_points
                loss_bbox = self.loss_bbox(
                        pos_pred_xy_offset,
                        pos_target_xy_offset,
                        pred_decode=pos_decoded_bbox_preds,
                        targets_decode=pos_decoded_target_preds,
                        weight=pos_centerness_targets,
                        avg_factor=centerness_denorm)
            else:
                loss_bbox = self.loss_bbox(
                    pos_decoded_bbox_preds,
                    pos_decoded_target_preds,
                    weight=pos_centerness_targets,
                    avg_factor=centerness_denorm)

            if self.separate_angle:
                loss_angle = self.loss_angle(
                    pos_angle_preds, pos_angle_targets, avg_factor=num_pos)
            loss_centerness = self.loss_centerness(
                pos_centerness, pos_centerness_targets, avg_factor=num_pos)
        else:
            loss_bbox = pos_bbox_preds.sum()
            loss_centerness = pos_centerness.sum()
            if self.separate_angle:
                loss_angle = pos_angle_preds.sum()

        if not self.gaucho_encoding and self.separate_angle:
            return dict(
                loss_cls=loss_cls,
                loss_bbox=loss_bbox,
                loss_angle=loss_angle,
                loss_centerness=loss_centerness)
        else:
            return dict(
                loss_cls=loss_cls,
                loss_bbox=loss_bbox,
                loss_centerness=loss_centerness)

    def _get_bboxes_single(self,
                           cls_scores,
                           bbox_preds,
                           angle_preds,
                           centernesses,
                           mlvl_points,
                           img_shape,
                           scale_factor,
                           cfg,
                           rescale=False):
        """Transform outputs for a single batch item into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for a single scale level
                Has shape (num_points * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for a single scale
                level with shape (num_points * 4, H, W).
            angle_preds (list[Tensor]): Box angle for a single scale level \
                with shape (N, num_points * 1, H, W).
            centernesses (list[Tensor]): Centerness for a single scale level
                with shape (num_points * 1, H, W).
            mlvl_points (list[Tensor]): Box reference for a single scale level
                with shape (num_total_points, 4).
            img_shape (tuple[int]): Shape of the input image,
                (height, width, 3).
            scale_factor (ndarray): Scale factor of the image arrange as
                (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.

        Returns:
            Tensor: Labeled boxes in shape (n, 6), where the first 5 columns
                are bounding box positions (x, y, w, h, angle) and the
                6-th column is a score between 0 and 1.
        """
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_points)
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_centerness = []
        for cls_score, bbox_pred, angle_pred, centerness, points in zip(
                cls_scores, bbox_preds, angle_preds, centernesses,
                mlvl_points):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            scores = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels).sigmoid()
            centerness = centerness.permute(1, 2, 0).reshape(-1).sigmoid()

            if self.gaucho_encoding:
                bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 5)
            else:
                bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
                angle_pred = angle_pred.permute(1, 2, 0).reshape(-1, 1)
                bbox_pred = torch.cat([bbox_pred, angle_pred], dim=1)
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = (scores * centerness[:, None]).max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                centerness = centerness[topk_inds]
            
            if self.gaucho_encoding:
                bboxes = decode_gaucho_offset(points, bbox_pred, self.angle_version, to_obb=True)
            else:
                bboxes = self.bbox_coder.decode(
                    points, bbox_pred, max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_centerness.append(centerness)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            scale_factor = mlvl_bboxes.new_tensor(scale_factor)
            mlvl_bboxes[..., :4] = mlvl_bboxes[..., :4] / scale_factor
        mlvl_scores = torch.cat(mlvl_scores)
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
        # BG cat_id: num_class
        mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)
        mlvl_centerness = torch.cat(mlvl_centerness)
        det_bboxes, det_labels = multiclass_nms_rotated(
            mlvl_bboxes,
            mlvl_scores,
            cfg.score_thr,
            cfg.nms,
            cfg.max_per_img,
            score_factors=mlvl_centerness)
        return det_bboxes, det_labels
