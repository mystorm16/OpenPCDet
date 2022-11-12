import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils import box_coder_utils, common_utils, loss_utils
from ..model_utils.model_nms_utils import class_agnostic_nms
from .target_assigner.proposal_target_layer import ProposalTargetLayer


class RoIHeadTemplate(nn.Module):
    def __init__(self, num_class, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.box_coder = getattr(box_coder_utils, self.model_cfg.TARGET_CONFIG.BOX_CODER)(
            **self.model_cfg.TARGET_CONFIG.get('BOX_CODER_CONFIG', {})
        )
        self.proposal_target_layer = ProposalTargetLayer(roi_sampler_cfg=self.model_cfg.TARGET_CONFIG)
        self.build_losses(self.model_cfg.LOSS_CONFIG)
        self.forward_ret_dict = None

    def build_losses(self, losses_cfg):
        self.add_module(
            'reg_loss_func',
            loss_utils.WeightedSmoothL1Loss(code_weights=losses_cfg.LOSS_WEIGHTS['code_weights'])
        )

    def make_fc_layers(self, input_channels, output_channels, fc_list):
        fc_layers = []
        pre_channel = input_channels
        for k in range(0, fc_list.__len__()):
            fc_layers.extend([
                nn.Conv1d(pre_channel, fc_list[k], kernel_size=1, bias=False),
                nn.BatchNorm1d(fc_list[k]),
                nn.ReLU()
            ])
            pre_channel = fc_list[k]
            if self.model_cfg.DP_RATIO >= 0 and k == 0:
                fc_layers.append(nn.Dropout(self.model_cfg.DP_RATIO))
        fc_layers.append(nn.Conv1d(pre_channel, output_channels, kernel_size=1, bias=True))
        fc_layers = nn.Sequential(*fc_layers)
        return fc_layers

    @torch.no_grad()
    # 从第一阶段每个点生成的proposal框里 通过NMS选取若干个ROI框
    def proposal_layer(self, batch_dict, nms_config):
        """
        Args:
            batch_dict:
                batch_size:
                batch_cls_preds: (B, num_boxes, num_classes | 1) or (N1+N2+..., num_classes | 1)
                batch_box_preds: (B, num_boxes, 7+C) or (N1+N2+..., 7+C)
                cls_preds_normalized: indicate whether batch_cls_preds is normalized
                batch_index: optional (N1+N2+...)
            nms_config:

        Returns:
            batch_dict:
                rois: (B, num_rois, 7+C)    包含了每个roi在3D空间的位置和大小
                roi_scores: (B, num_rois)   包含了每个roi置信度分数
                roi_labels: (B, num_rois)   包含了每个roi对应的类别

        """
        if batch_dict.get('rois', None) is not None:
            return batch_dict
        # 得到batch_size
        batch_size = batch_dict['batch_size']
        # 得到每一批数据的box预测结果 shape :（batch * 16384, 7）
        batch_box_preds = batch_dict['batch_box_preds']
        # 得到每一批数据的cls预测结果 shape :（batch * 16384, 3）
        batch_cls_preds = batch_dict['batch_cls_preds']
        # 用0初始化所有的rois          shape : （batch, 512, 7） 训练时为512个roi，测试时为100个roi
        rois = batch_box_preds.new_zeros((batch_size, nms_config.NMS_POST_MAXSIZE, batch_box_preds.shape[-1]))
        # 用0初始化所有的roi_score     shape : （batch, 512） 训练时为512个roi，测试时为100个roi
        roi_scores = batch_box_preds.new_zeros((batch_size, nms_config.NMS_POST_MAXSIZE))
        # 用0初始化所有的roi_labels    shape : （batch, 512） 训练时为512个roi，测试时为100个roi
        roi_labels = batch_box_preds.new_zeros((batch_size, nms_config.NMS_POST_MAXSIZE), dtype=torch.long)
        # 逐帧计算rois
        for index in range(batch_size):
            if batch_dict.get('batch_index', None) is not None:
                assert batch_cls_preds.shape.__len__() == 2
                # 得到所有属于当前帧点的mask
                batch_mask = (batch_dict['batch_index'] == index)
            else:
                assert batch_dict['batch_cls_preds'].shape.__len__() == 3
                batch_mask = index
            # 得到当前帧点的box预测结果和对应的cls预测结果 box ：（16384, 7）; cls ：（16384, 3）
            box_preds = batch_box_preds[batch_mask]
            cls_preds = batch_cls_preds[batch_mask]
            # 取出每个点类别预测的最大数值和最大数值所对应的索引 cur_roi_scores: (16384, )
            cur_roi_scores, cur_roi_labels = torch.max(cls_preds, dim=1)
            if nms_config.MULTI_CLASSES_NMS:
                raise NotImplementedError
            else:
                # 进行无类别的nms操作  selected为经过NMS操作后被留下来的box的索引，selected_scores为被留下来box的最大类别预测分数
                selected, selected_scores = class_agnostic_nms(
                    box_scores=cur_roi_scores, box_preds=box_preds, nms_config=nms_config
                )
            # 从所有预测结果中选取经过nms操作后得到的box存入roi中
            rois[index, :len(selected), :] = box_preds[selected]
            # 从所有预测结果中选取经过nms操作后得到的box对应类别分数存入roi中
            roi_scores[index, :len(selected)] = cur_roi_scores[selected]
            # 从所有预测结果中选取经过nms操作后得到的box对应类别存入roi中
            roi_labels[index, :len(selected)] = cur_roi_labels[selected]
        # 处理结果，成字典形式并返回
        batch_dict['rois'] = rois
        batch_dict['roi_scores'] = roi_scores
        batch_dict['roi_labels'] = roi_labels + 1
        batch_dict['has_class_labels'] = True if batch_cls_preds.shape[-1] > 1 else False
        batch_dict.pop('batch_index', None)
        return batch_dict

    # ①根据第一阶段输出结果整理出正负样本为1：1送给第二阶段网络。
    # ②把gt转到ccs系
    def assign_targets(self, batch_dict):
        batch_size = batch_dict['batch_size']
        with torch.no_grad():
            """
                targets_dict={
                'rois': batch_rois,                 roi的box的7个参数                    shape（batch， 128, 7）
                'gt_of_rois': batch_gt_of_rois,     roi对应的GTbox的8个参数，包含类别      shape（batch， 128, 8）
                'gt_iou_of_rois': batch_roi_ious,   roi个对应GTbox的最大iou数值           shape（batch， 128）
                'roi_scores': batch_roi_scores,     roi box的类别预测分数                shape（batch， 128）
                'roi_labels': batch_roi_labels,     roi box的类别预测结果              shape（batch， 128）
                'reg_valid_mask': reg_valid_mask,   需要计算回归损失的roi            shape（batch， 128）
                'rcnn_cls_labels': batch_cls_labels 计算前背景损失的roi             shape（batch， 128）
                }
            """
            # 完成128个proposal的选取和正负proposal分配
            targets_dict = self.proposal_target_layer.forward(batch_dict)

        # 完成匹配到的GT转换到CCS坐标系
        rois = targets_dict['rois']  # (B, N, 7 + C)
        gt_of_rois = targets_dict['gt_of_rois']  # (B, N, 7 + C + 1)
        targets_dict['gt_of_rois_src'] = gt_of_rois.clone().detach()

        # canonical transformation 为了计算gt和pred的loss
        # 进行canonical transformation变换，需要roi的xyz在点云中的坐标位置转换到以自身中心为原点
        roi_center = rois[:, :, 0:3]
        # 将heading的数值，由-pi-pi转到0-2pi中  弧度
        roi_ry = rois[:, :, 6] % (2 * np.pi)
        # 计算在经过canonical transformation变换后GT相对于以roi中心的x，y，z偏移量
        gt_of_rois[:, :, 0:3] = gt_of_rois[:, :, 0:3] - roi_center
        # 计算GT和roi的heading偏移量
        gt_of_rois[:, :, 6] = gt_of_rois[:, :, 6] - roi_ry

        # transfer LiDAR coords to local coords
        # 上面完成了点云坐标系中的GT到roi坐标系的xyz方向的偏移，下面完成点云坐标系中的GT到roi坐标系的角度旋转，
        # 其中点云坐标系，x向前，y向右，z向上；roi坐标系中，x朝车头方向，y与x垂直，z向上
        gt_of_rois = common_utils.rotate_points_along_z(
            points=gt_of_rois.view(-1, 1, gt_of_rois.shape[-1]), angle=-roi_ry.view(-1)
        ).view(batch_size, -1, gt_of_rois.shape[-1])

        # flip orientation if rois have opposite orientation
        heading_label = gt_of_rois[:, :, 6] % (2 * np.pi)  # 0 ~ 2pi
        opposite_flag = (heading_label > np.pi * 0.5) & (heading_label < np.pi * 1.5)
        heading_label[opposite_flag] = (heading_label[opposite_flag] + np.pi) % (2 * np.pi)  # (0 ~ pi/2, 3pi/2 ~ 2pi)
        flag = heading_label > np.pi
        heading_label[flag] = heading_label[flag] - np.pi * 2  # (-pi/2, pi/2)
        # 在3D的iou计算中，如果两个box的iou大于0.55，那么他们的角度偏差只会在-45度到45度之间
        heading_label = torch.clamp(heading_label, min=-np.pi / 2, max=np.pi / 2)

        gt_of_rois[:, :, 6] = heading_label
        targets_dict['gt_of_rois'] = gt_of_rois
        return targets_dict

    def get_box_reg_layer_loss(self, forward_ret_dict):
        loss_cfgs = self.model_cfg.LOSS_CONFIG
        code_size = self.box_coder.code_size
        reg_valid_mask = forward_ret_dict['reg_valid_mask'].view(-1)
        gt_boxes3d_ct = forward_ret_dict['gt_of_rois'][..., 0:code_size]
        gt_of_rois_src = forward_ret_dict['gt_of_rois_src'][..., 0:code_size].view(-1, code_size)
        rcnn_reg = forward_ret_dict['rcnn_reg']  # (rcnn_batch_size, C)
        roi_boxes3d = forward_ret_dict['rois']
        rcnn_batch_size = gt_boxes3d_ct.view(-1, code_size).shape[0]

        fg_mask = (reg_valid_mask > 0)
        fg_sum = fg_mask.long().sum().item()

        tb_dict = {}

        if loss_cfgs.REG_LOSS == 'smooth-l1':
            rois_anchor = roi_boxes3d.clone().detach().view(-1, code_size)
            rois_anchor[:, 0:3] = 0
            rois_anchor[:, 6] = 0
            reg_targets = self.box_coder.encode_torch(
                gt_boxes3d_ct.view(rcnn_batch_size, code_size), rois_anchor
            )

            rcnn_loss_reg = self.reg_loss_func(
                rcnn_reg.view(rcnn_batch_size, -1).unsqueeze(dim=0),
                reg_targets.unsqueeze(dim=0),
            )  # [B, M, 7]
            rcnn_loss_reg = (rcnn_loss_reg.view(rcnn_batch_size, -1) * fg_mask.unsqueeze(dim=-1).float()).sum() / max(fg_sum, 1)
            rcnn_loss_reg = rcnn_loss_reg * loss_cfgs.LOSS_WEIGHTS['rcnn_reg_weight']
            tb_dict['rcnn_loss_reg'] = rcnn_loss_reg.item()

            if loss_cfgs.CORNER_LOSS_REGULARIZATION and fg_sum > 0:
                # TODO: NEED to BE CHECK
                fg_rcnn_reg = rcnn_reg.view(rcnn_batch_size, -1)[fg_mask]
                fg_roi_boxes3d = roi_boxes3d.view(-1, code_size)[fg_mask]

                fg_roi_boxes3d = fg_roi_boxes3d.view(1, -1, code_size)
                batch_anchors = fg_roi_boxes3d.clone().detach()
                roi_ry = fg_roi_boxes3d[:, :, 6].view(-1)
                roi_xyz = fg_roi_boxes3d[:, :, 0:3].view(-1, 3)
                batch_anchors[:, :, 0:3] = 0
                rcnn_boxes3d = self.box_coder.decode_torch(
                    fg_rcnn_reg.view(batch_anchors.shape[0], -1, code_size), batch_anchors
                ).view(-1, code_size)

                rcnn_boxes3d = common_utils.rotate_points_along_z(
                    rcnn_boxes3d.unsqueeze(dim=1), roi_ry
                ).squeeze(dim=1)
                rcnn_boxes3d[:, 0:3] += roi_xyz

                loss_corner = loss_utils.get_corner_loss_lidar(
                    rcnn_boxes3d[:, 0:7],
                    gt_of_rois_src[fg_mask][:, 0:7]
                )
                loss_corner = loss_corner.mean()
                loss_corner = loss_corner * loss_cfgs.LOSS_WEIGHTS['rcnn_corner_weight']

                rcnn_loss_reg += loss_corner
                tb_dict['rcnn_loss_corner'] = loss_corner.item()
        else:
            raise NotImplementedError

        return rcnn_loss_reg, tb_dict

    def get_box_cls_layer_loss(self, forward_ret_dict):
        loss_cfgs = self.model_cfg.LOSS_CONFIG
        rcnn_cls = forward_ret_dict['rcnn_cls']
        rcnn_cls_labels = forward_ret_dict['rcnn_cls_labels'].view(-1)
        if loss_cfgs.CLS_LOSS == 'BinaryCrossEntropy':
            rcnn_cls_flat = rcnn_cls.view(-1)
            batch_loss_cls = F.binary_cross_entropy(torch.sigmoid(rcnn_cls_flat), rcnn_cls_labels.float(), reduction='none')
            cls_valid_mask = (rcnn_cls_labels >= 0).float()
            rcnn_loss_cls = (batch_loss_cls * cls_valid_mask).sum() / torch.clamp(cls_valid_mask.sum(), min=1.0)
        elif loss_cfgs.CLS_LOSS == 'CrossEntropy':
            batch_loss_cls = F.cross_entropy(rcnn_cls, rcnn_cls_labels, reduction='none', ignore_index=-1)
            cls_valid_mask = (rcnn_cls_labels >= 0).float()
            rcnn_loss_cls = (batch_loss_cls * cls_valid_mask).sum() / torch.clamp(cls_valid_mask.sum(), min=1.0)
        else:
            raise NotImplementedError

        rcnn_loss_cls = rcnn_loss_cls * loss_cfgs.LOSS_WEIGHTS['rcnn_cls_weight']
        tb_dict = {'rcnn_loss_cls': rcnn_loss_cls.item()}
        return rcnn_loss_cls, tb_dict

    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict
        rcnn_loss = 0
        rcnn_loss_cls, cls_tb_dict = self.get_box_cls_layer_loss(self.forward_ret_dict)
        rcnn_loss += rcnn_loss_cls
        tb_dict.update(cls_tb_dict)

        rcnn_loss_reg, reg_tb_dict = self.get_box_reg_layer_loss(self.forward_ret_dict)
        rcnn_loss += rcnn_loss_reg
        tb_dict.update(reg_tb_dict)
        tb_dict['rcnn_loss'] = rcnn_loss.item()
        return rcnn_loss, tb_dict

    def generate_predicted_boxes(self, batch_size, rois, cls_preds, box_preds):
        """
        Args:
            batch_size:
            rois: (B, N, 7)
            cls_preds: (BN, num_class)
            box_preds: (BN, code_size)

        Returns:

        """
        code_size = self.box_coder.code_size
        # batch_cls_preds: (B, N, num_class or 1)
        batch_cls_preds = cls_preds.view(batch_size, -1, cls_preds.shape[-1])
        batch_box_preds = box_preds.view(batch_size, -1, code_size)

        roi_ry = rois[:, :, 6].view(-1)
        roi_xyz = rois[:, :, 0:3].view(-1, 3)
        local_rois = rois.clone().detach()
        local_rois[:, :, 0:3] = 0

        batch_box_preds = self.box_coder.decode_torch(batch_box_preds, local_rois).view(-1, code_size)

        batch_box_preds = common_utils.rotate_points_along_z(
            batch_box_preds.unsqueeze(dim=1), roi_ry
        ).squeeze(dim=1)
        batch_box_preds[:, 0:3] += roi_xyz
        batch_box_preds = batch_box_preds.view(batch_size, -1, code_size)
        return batch_cls_preds, batch_box_preds
