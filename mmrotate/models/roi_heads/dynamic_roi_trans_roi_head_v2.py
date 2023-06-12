# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta

import torch
from mmcv.runner import BaseModule, ModuleList
from mmdet.core import bbox2roi

from mmrotate.core import (build_assigner, build_sampler, obb2xyxy,
                           rbbox2result, rbbox2roi)
from ..builder import ROTATED_HEADS, build_head, build_roi_extractor

import numpy as np
import os

EPS = 1e-15

@ROTATED_HEADS.register_module()
class DynamicRoITransRoIHeadv2(BaseModule, metaclass=ABCMeta):
    """RoI Trans cascade roi head including one bbox head.

    Args:
        num_stages (int): number of cascade stages.
        stage_loss_weights (list[float]): loss weights of cascade stages.
        bbox_roi_extractor (dict, optional): Config of ``bbox_roi_extractor``.
        bbox_head (dict, optional): Config of ``bbox_head``.
        shared_head (dict, optional): Config of ``shared_head``.
        train_cfg (dict, optional): Config of train.
        test_cfg (dict, optional): Config of test.
        pretrained (str, optional): Path of pretrained weight.
        version (str, optional): Angle representations. Defaults to 'oc'.
        init_cfg (dict, optional): Config of initialization.
    """

    def __init__(self,
                 num_stages,
                 stage_loss_weights,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 shared_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 version='oc',
                 init_cfg=None):
        assert bbox_roi_extractor is not None
        assert bbox_head is not None
        assert shared_head is None, \
            'Shared head is not supported in Cascade RCNN anymore'

        super(DynamicRoITransRoIHeadv2, self).__init__(init_cfg)
        self.num_stages = num_stages
        self.stage_loss_weights = stage_loss_weights
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.pretrained = pretrained
        self.version = version

        if bbox_head is not None:
            self.init_bbox_head(bbox_roi_extractor, bbox_head)

        self.init_assigner_sampler()

        self.with_bbox = True if self.bbox_head is not None else False
        self.iou_history = [[], []]
        self.beta_history = [[], []]

    def init_bbox_head(self, bbox_roi_extractor, bbox_head):
        """Initialize box head and box roi extractor.

        Args:
            bbox_roi_extractor (dict): Config of box roi extractor.
            bbox_head (dict): Config of box in box head.
        """
        self.bbox_roi_extractor = ModuleList()
        self.bbox_head = ModuleList()
        if not isinstance(bbox_roi_extractor, list):
            bbox_roi_extractor = [
                bbox_roi_extractor for _ in range(self.num_stages)
            ]
        if not isinstance(bbox_head, list):
            bbox_head = [bbox_head for _ in range(self.num_stages)]
        assert len(bbox_roi_extractor) == len(bbox_head) == self.num_stages
        for roi_extractor, head in zip(bbox_roi_extractor, bbox_head):
            self.bbox_roi_extractor.append(build_roi_extractor(roi_extractor))
            self.bbox_head.append(build_head(head))

    def init_assigner_sampler(self):
        """Initialize assigner and sampler for each stage."""
        self.bbox_assigner = []
        self.bbox_sampler = []
        if self.train_cfg is not None:
            for idx, rcnn_train_cfg in enumerate(self.train_cfg):
                self.bbox_assigner.append(
                    build_assigner(rcnn_train_cfg.assigner))
                self.current_stage = idx
                self.bbox_sampler.append(
                    build_sampler(rcnn_train_cfg.sampler, context=self))

    def forward_dummy(self, x, proposals):
        """Dummy forward function.

        Args:
            x (list[Tensors]): list of multi-level img features.
            proposals (list[Tensors]): list of region proposals.

        Returns:
            list[Tensors]: list of region of interest.
        """
        # bbox head
        outs = ()
        rois = bbox2roi([proposals])
        if self.with_bbox:
            for i in range(self.num_stages):
                if i > 0:
                    rois = rbbox2roi([proposals])
                bbox_results = self._bbox_forward(i, x, rois)
                proposals = torch.randn(1000, 6).to(proposals.device)
                outs = outs + (bbox_results['cls_score'],
                               bbox_results['bbox_pred'])
        return outs
    
    def _bbox_forward(self, stage, x, rois):
        """Box head forward function used in both training and testing.

        Args:
            x (list[Tensor]): list of multi-level img features.
            rois (list[Tensors]): list of region of interests.

        Returns:
            dict[str, Tensor]: a dictionary of bbox_results.
        """
        bbox_roi_extractor = self.bbox_roi_extractor[stage]
        bbox_head = self.bbox_head[stage]
        bbox_feats = bbox_roi_extractor(x[:bbox_roi_extractor.num_inputs],
                                        rois)
        # do not support caffe_c4 model anymore
        cls_score, bbox_pred = bbox_head(bbox_feats)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        '''
            return cls_score, bbox_pred
        '''
        return bbox_results

    # def bbox2roi(bbox_list):
    #     """Convert a list of bboxes to roi format.

    #     Args:
    #         bbox_list (list[Tensor]): a list of bboxes corresponding to a batch
    #             of images.

    #     Returns:
    #         Tensor: shape (n, 5), [batch_ind, x1, y1, x2, y2]
    #     """
    #     rois_list = []
    #     for img_id, bboxes in enumerate(bbox_list):
    #         if bboxes.size(0) > 0:
    #             img_inds = bboxes.new_full((bboxes.size(0), 1), img_id)
    #             rois = torch.cat([img_inds, bboxes[:, :4]], dim=-1)
    #         else:
    #             rois = bboxes.new_zeros((0, 5))
    #         rois_list.append(rois)
    #     rois = torch.cat(rois_list, 0)
    #     return rois
    
    # def rbbox2roi(bbox_list):
    #     """Convert a list of bboxes to roi format.

    #     Args:
    #         bbox_list (list[Tensor]): a list of bboxes corresponding to a batch
    #             of images.

    #     Returns:
    #         Tensor: shape (n, 6), [batch_ind, cx, cy, w, h, a]
    #     """
    #     rois_list = []
    #     for img_id, bboxes in enumerate(bbox_list):
    #         if bboxes.size(0) > 0:
    #             img_inds = bboxes.new_full((bboxes.size(0), 1), img_id)
    #             rois = torch.cat([img_inds, bboxes[:, :5]], dim=-1)
    #         else:
    #             rois = bboxes.new_zeros((0, 6))
    #         rois_list.append(rois)
    #     rois = torch.cat(rois_list, 0)
    #     return rois

    def _bbox_forward_train(self, stage, x, sampling_results, gt_bboxes,
                            gt_labels, rcnn_train_cfg, img_metas, is_dynamic):
        """Run forward function and calculate loss for box head in training.

        Args:
            x (list[Tensor]): list of multi-level img features.
            sampling_results (list[Tensor]): list of sampling results.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 5) in [cx, cy, w, h, a] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.

        Returns:
            dict[str, Tensor]: a dictionary of bbox_results.
        """
        if stage == 0:
            rois = bbox2roi([res.bboxes for res in sampling_results])
            '''
            Returns:
                Tensor: shape (n, 5), [batch_ind, x1, y1, x2, y2]
            '''
        else:
            rois = rbbox2roi([res.bboxes for res in sampling_results])
            '''
            Returns:
                Tensor: shape (n, 6), [batch_ind, cx, cy, w, h, a]
            '''
        bbox_results = self._bbox_forward(stage, x, rois)
        bbox_targets = self.bbox_head[stage].get_targets(
            sampling_results, gt_bboxes, gt_labels, rcnn_train_cfg)
        
        if (is_dynamic):
            # breakpoint()
            num_imgs = len(img_metas)
            pos_inds = bbox_targets[3][:, 0].nonzero().squeeze(1)
            num_pos = len(pos_inds)
            cur_target = bbox_targets[2][pos_inds, :2].abs().mean(dim=1)
            beta_topk = min(self.train_cfg[stage].dynamic_rcnn.beta_topk * num_imgs,
                            num_pos)
            cur_target = torch.kthvalue(cur_target, beta_topk)[0].item()
            self.beta_history[stage].append(cur_target)
        
        loss_bbox = self.bbox_head[stage].loss(bbox_results['cls_score'],
                                               bbox_results['bbox_pred'], rois,
                                               *bbox_targets)

        bbox_results.update(
            loss_bbox=loss_bbox, rois=rois, bbox_targets=bbox_targets)
        return bbox_results

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            proposals (list[Tensors]): list of region proposals.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 5) in [cx, cy, w, h, a] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task. Always
                set to None.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        losses = dict()
        for i in range(self.num_stages):
            self.current_stage = i
            rcnn_train_cfg = self.train_cfg[i]
            lw = self.stage_loss_weights[i]

            # assign gts and sample proposals
            sampling_results = []
            cur_iou = [[], []]
            
            if self.with_bbox:
                bbox_assigner = self.bbox_assigner[i]
                bbox_sampler = self.bbox_sampler[i]
                num_imgs = len(img_metas)
                if gt_bboxes_ignore is None:
                    gt_bboxes_ignore = [None for _ in range(num_imgs)]

                for j in range(num_imgs):
                    if i == 0:
                        gt_tmp_bboxes = obb2xyxy(gt_bboxes[j], self.version)
                    else:
                        gt_tmp_bboxes = gt_bboxes[j]
                    assign_result = bbox_assigner.assign(
                        proposal_list[j], gt_tmp_bboxes, gt_bboxes_ignore[j],
                        gt_labels[j])
                    sampling_result = bbox_sampler.sample(
                        assign_result,
                        proposal_list[j],
                        gt_tmp_bboxes,
                        gt_labels[j],
                        feats=[lvl_feat[j][None] for lvl_feat in x])

                    if 'dynamic_rcnn' in self.train_cfg[i]:
                        iou_topk = min(self.train_cfg[i].dynamic_rcnn.iou_topk,
                                    len(assign_result.max_overlaps))
                        ious, _ = torch.topk(assign_result.max_overlaps, iou_topk)
                        cur_iou[i].append(ious[-1].item())
                    
                    if gt_bboxes[j].numel() == 0:
                        sampling_result.pos_gt_bboxes = gt_bboxes[j].new(
                            (0, gt_bboxes[0].size(-1))).zero_()
                    else:
                        sampling_result.pos_gt_bboxes = \
                            gt_bboxes[j][
                                sampling_result.pos_assigned_gt_inds, :]

                    sampling_results.append(sampling_result)

                if 'dynamic_rcnn' in self.train_cfg[i]:
                    cur_iou = np.mean(cur_iou)
                    self.iou_history[i].append(cur_iou)
            # bbox head forward and loss
            bbox_results = self._bbox_forward_train(i, x, sampling_results,
                                                    gt_bboxes, gt_labels,
                                                    rcnn_train_cfg,
                                                    img_metas,
                                                    is_dynamic = 'dynamic_rcnn' in self.train_cfg[i])

            for name, value in bbox_results['loss_bbox'].items():
                losses[f's{i}.{name}'] = (
                    value * lw if 'loss' in name else value)
            
            # refine bboxes
            if i < self.num_stages - 1:
                pos_is_gts = [res.pos_is_gt for res in sampling_results]
                # bbox_targets is a tuple
                roi_labels = bbox_results['bbox_targets'][0]
                with torch.no_grad():
                    cls_score = bbox_results['cls_score']
                    if self.bbox_head[i].custom_activation:
                        cls_score = self.bbox_head[i].loss_cls.get_activation(
                            cls_score)
                    roi_labels = torch.where(
                        roi_labels == self.bbox_head[i].num_classes,
                        cls_score[:, :-1].argmax(1), roi_labels)
                    # https://github.com/open-mmlab/mmdetection/issues/6455
                    proposal_list = self.bbox_head[i].refine_bboxes(
                        bbox_results['rois'], roi_labels,
                        bbox_results['bbox_pred'], pos_is_gts, img_metas)
                    
            if 'dynamic_rcnn' in self.train_cfg[i]:
                update_iter_interval = self.train_cfg[i].dynamic_rcnn.update_iter_interval
                if len(self.iou_history[i]) % update_iter_interval == 0:
                    new_iou_thr, new_beta = self.update_hyperparameters(i)
                    # Writing data to a file
                    f = open(os.path.join(self.train_cfg[i].dynamic_rcnn.save_dir, f"update_hyperparameters_{i}.txt"), 'a')
                    f.write(f"{str(new_iou_thr)}, {str(new_beta)} \n")
                    f.close()
        return losses

    def simple_test(self, x, proposal_list, img_metas, rescale=False):
        """Test without augmentation.

        Args:
            x (list[Tensor]): list of multi-level img features.
            proposal_list (list[Tensors]): list of region proposals.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
            rescale (bool): If True, return boxes in original image space.
                Default: False.

        Returns:
            dict[str, Tensor]: a dictionary of bbox_results.
        """
        assert self.with_bbox, 'Bbox head must be implemented.'
        num_imgs = len(proposal_list)
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        # "ms" in variable names means multi-stage
        ms_bbox_result = {}
        ms_scores = []
        rcnn_test_cfg = self.test_cfg

        rois = bbox2roi(proposal_list)
        for i in range(self.num_stages):
            bbox_results = self._bbox_forward(i, x, rois)

            # split batch bbox prediction back to each image
            cls_score = bbox_results['cls_score']
            bbox_pred = bbox_results['bbox_pred']
            num_proposals_per_img = tuple(
                len(proposals) for proposals in proposal_list)
            rois = rois.split(num_proposals_per_img, 0)
            cls_score = cls_score.split(num_proposals_per_img, 0)
            if isinstance(bbox_pred, torch.Tensor):
                bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            else:
                bbox_pred = self.bbox_head[i].bbox_pred_split(
                    bbox_pred, num_proposals_per_img)
            ms_scores.append(cls_score)

            if i < self.num_stages - 1:
                if self.bbox_head[i].custom_activation:
                    cls_score = [
                        self.bbox_head[i].loss_cls.get_activation(s)
                        for s in cls_score
                    ]
                bbox_label = [s[:, :-1].argmax(dim=1) for s in cls_score]
                rois = torch.cat([
                    self.bbox_head[i].regress_by_class(rois[j], bbox_label[j],
                                                       bbox_pred[j],
                                                       img_metas[j])
                    for j in range(num_imgs)
                ])

        # average scores of each image by stages
        cls_score = [
            sum([score[i] for score in ms_scores]) / float(len(ms_scores))
            for i in range(num_imgs)
        ]

        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        for i in range(num_imgs):
            det_bbox, det_label = self.bbox_head[-1].get_bboxes(
                rois[i],
                cls_score[i],
                bbox_pred[i],
                img_shapes[i],
                scale_factors[i],
                rescale=rescale,
                cfg=rcnn_test_cfg)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)

        bbox_results = [
            rbbox2result(det_bboxes[i], det_labels[i],
                         self.bbox_head[-1].num_classes)
            for i in range(num_imgs)
        ]
        ms_bbox_result['ensemble'] = bbox_results
        results = ms_bbox_result['ensemble']

        return results

    def aug_test(self, features, proposal_list, img_metas, rescale=False):
        """Test with augmentations."""
        raise NotImplementedError
    
    def update_hyperparameters(self, stage):
        """Update hyperparameters like IoU thresholds for assigner and beta for
        SmoothL1 loss based on the training statistics.

        Returns:
            tuple[float]: the updated ``iou_thr`` and ``beta``.
        """
        new_iou_thr = max(self.train_cfg[stage].dynamic_rcnn.initial_iou,
                          np.mean(self.iou_history[stage]))
        self.iou_history[stage] = []
        # Just update positive thr
        self.bbox_assigner[stage].pos_iou_thr = new_iou_thr
        self.bbox_assigner[stage].neg_iou_thr = new_iou_thr
        self.bbox_assigner[stage].min_pos_iou = new_iou_thr

        if (str(self.bbox_head[stage].loss_bbox) == "SmoothL1Loss()"):
            if (np.median(self.beta_history[stage]) < EPS):
                # avoid 0 or too small value for new_beta
                new_beta = self.bbox_head[stage].loss_bbox.beta
            else:
                new_beta = min(self.train_cfg[stage].dynamic_rcnn.initial_beta,
                            np.median(self.beta_history[stage]))
            self.beta_history[stage] = []
            self.bbox_head[stage].loss_bbox.beta = new_beta
        else:
            new_beta = None
        return new_iou_thr, new_beta
