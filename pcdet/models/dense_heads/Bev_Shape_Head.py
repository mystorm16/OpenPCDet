import copy
import time
from numba import jit

import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_
from ..model_utils import model_nms_utils
from ..model_utils import centernet_utils
from ..model_utils.centernet_utils import gaussian2D, sy_draw_gaussian_to_heatmap
from ...utils import loss_utils
import seaborn as sns;
from tools.visual_utils.open3d_vis_utils import draw_scenes, draw_scenes_voxel_a, draw_scenes_voxel_b


sns.set()
import matplotlib.pyplot as plt


class SeparateHead(nn.Module):
    def __init__(self, input_channels, sep_head_dict, init_bias=-2.19, use_bias=False):
        super().__init__()
        self.sep_head_dict = sep_head_dict

        for cur_name in self.sep_head_dict:  # 搭建center centerz dim rot hm的输出头网络,都是conv2d(64 64)卷一下然后接全连接
            output_channels = self.sep_head_dict[cur_name]['out_channels']
            num_conv = self.sep_head_dict[cur_name]['num_conv']

            fc_list = []
            for k in range(num_conv - 1):
                fc_list.append(nn.Sequential(
                    nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=1, padding=1, bias=use_bias),
                    nn.BatchNorm2d(input_channels),
                    nn.ReLU()
                ))
            fc_list.append(nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=True))
            fc = nn.Sequential(*fc_list)
            if 'hm' in cur_name:
                fc[-1].bias.data.fill_(init_bias)
            else:
                for m in fc.modules():
                    if isinstance(m, nn.Conv2d):
                        kaiming_normal_(m.weight.data)
                        if hasattr(m, "bias") and m.bias is not None:
                            nn.init.constant_(m.bias, 0)

            self.__setattr__(cur_name, fc)

    def forward(self, x):
        ret_dict = {}
        for cur_name in self.sep_head_dict:
            ret_dict[cur_name] = self.__getattr__(cur_name)(x)  # 输入2D feature map进输出头(64 64)(64 x)进行预测

        return ret_dict


class BevShapeHead(nn.Module):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range, voxel_size,
                 predict_boxes_when_training=True):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.grid_size = grid_size
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
        self.feature_map_stride = self.model_cfg.TARGET_ASSIGNER_CONFIG.get('FEATURE_MAP_STRIDE', None)

        self.class_names = class_names
        self.class_names_each_head = []
        self.class_id_mapping_each_head = []

        for cur_class_names in self.model_cfg.CLASS_NAMES_EACH_HEAD:
            self.class_names_each_head.append([x for x in cur_class_names if x in class_names])
            cur_class_id_mapping = torch.from_numpy(np.array(
                [self.class_names.index(x) for x in cur_class_names if x in class_names]
            )).cuda()
            self.class_id_mapping_each_head.append(cur_class_id_mapping)

        total_classes = sum([len(x) for x in self.class_names_each_head])
        assert total_classes == len(self.class_names), f'class_names_each_head={self.class_names_each_head}'

        self.shared_conv = nn.Sequential(
            nn.Conv2d(
                input_channels, self.model_cfg.SHARED_CONV_CHANNEL, 3, stride=1, padding=1,
                bias=self.model_cfg.get('USE_BIAS_BEFORE_NORM', False)
            ),
            nn.BatchNorm2d(self.model_cfg.SHARED_CONV_CHANNEL),
            nn.ReLU(),
        )

        self.heads_list = nn.ModuleList()
        self.separate_head_cfg = self.model_cfg.SEPARATE_HEAD_CFG
        for idx, cur_class_names in enumerate(self.class_names_each_head):
            cur_head_dict = copy.deepcopy(self.separate_head_cfg.HEAD_DICT)
            cur_head_dict['hm'] = dict(out_channels=len(cur_class_names), num_conv=self.model_cfg.NUM_HM_CONV)
            self.heads_list.append(
                SeparateHead(
                    input_channels=self.model_cfg.SHARED_CONV_CHANNEL,
                    sep_head_dict=cur_head_dict,
                    init_bias=-2.19,
                    use_bias=self.model_cfg.get('USE_BIAS_BEFORE_NORM', False)
                )
            )
        self.predict_boxes_when_training = predict_boxes_when_training
        self.forward_ret_dict = {}
        self.build_losses()

    def build_losses(self):
        self.add_module('hm_loss_func', loss_utils.FocalLossCenterNet())

    def assign_target_of_single_head(
            self, num_classes, gt_boxes, feature_map_size, feature_map_stride, num_max_objs=500,
            gaussian_overlap=0.1, min_radius=2
    ):
        """
        Args:
            gt_boxes: (N, 8)
            feature_map_size: (2), [x, y]

        Returns:

        """
        heatmap = gt_boxes.new_zeros(num_classes, feature_map_size[1], feature_map_size[0])
        ret_boxes = gt_boxes.new_zeros((num_max_objs, gt_boxes.shape[-1] - 1 + 1))
        inds = gt_boxes.new_zeros(num_max_objs).long()
        mask = gt_boxes.new_zeros(num_max_objs).long()

        x, y, z = gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2]
        coord_x = (x - self.point_cloud_range[0]) / self.voxel_size[0] / feature_map_stride
        coord_y = (y - self.point_cloud_range[1]) / self.voxel_size[1] / feature_map_stride
        coord_x = torch.clamp(coord_x, min=0,
                              max=feature_map_size[0] - 0.5)  # bugfixed: 1e-6 does not work for center.int()
        coord_y = torch.clamp(coord_y, min=0, max=feature_map_size[1] - 0.5)  #
        center = torch.cat((coord_x[:, None], coord_y[:, None]), dim=-1)
        center_int = center.int()
        center_int_float = center_int.float()

        dx, dy, dz = gt_boxes[:, 3], gt_boxes[:, 4], gt_boxes[:, 5]
        dx = dx / self.voxel_size[0] / feature_map_stride
        dy = dy / self.voxel_size[1] / feature_map_stride

        radius = centernet_utils.gaussian_radius(dx, dy, min_overlap=gaussian_overlap)
        radius = torch.clamp_min(radius.int(), min=min_radius)

        for k in range(min(num_max_objs, gt_boxes.shape[0])):
            if dx[k] <= 0 or dy[k] <= 0:
                continue

            if not (0 <= center_int[k][0] <= feature_map_size[0] and 0 <= center_int[k][1] <= feature_map_size[1]):
                continue

            cur_class_id = (gt_boxes[k, -1] - 1).long()
            centernet_utils.draw_gaussian_to_heatmap(heatmap[cur_class_id], center[k], radius[k].item())

            inds[k] = center_int[k, 1] * feature_map_size[0] + center_int[k, 0]
            mask[k] = 1

            ret_boxes[k, 0:2] = center[k] - center_int_float[k].float()
            ret_boxes[k, 2] = z[k]
            ret_boxes[k, 3:6] = gt_boxes[k, 3:6].log()
            ret_boxes[k, 6] = torch.cos(gt_boxes[k, 6])
            ret_boxes[k, 7] = torch.sin(gt_boxes[k, 6])
            if gt_boxes.shape[1] > 8:
                ret_boxes[k, 8:] = gt_boxes[k, 7:-1]

        return heatmap, ret_boxes, inds, mask

    def sy_assign_target_of_single_head(
            self, num_classes, feature_map_size, data_dict,
            bs_idx = None,
    ):
        """
        Args:
            gt_boxes: (N, 8)
            feature_map_size: (2), [x, y]
        Returns:
        """
        heatmap = torch.zeros(num_classes, feature_map_size[1], feature_map_size[0]).cuda()

        # 生成car heatmap 真值
        coord_c = data_dict['c_bm_spatial_features'][bs_idx].reshape(496, 432)
        coord_c = (coord_c == 1).nonzero(as_tuple=False)
        c_coord_x = coord_c[:, 0]
        c_coord_y = coord_c[:, 1]
        c_center = torch.cat((c_coord_x[:, None], c_coord_y[:, None]), dim=-1)
        radius = int(self.model_cfg.TARGET_ASSIGNER_CONFIG.C_RADIUS)  # 高斯半径
        heatmap = sy_draw_gaussian_to_heatmap(heatmap, c_center, radius, 0)

        # 生成ped heatmap 真值
        coord_p = data_dict['p_bm_spatial_features'][bs_idx].reshape(496, 432)
        coord_p = (coord_p == 1).nonzero(as_tuple=False)
        p_coord_x = coord_p[:, 0]
        p_coord_y = coord_p[:, 1]
        p_center = torch.cat((p_coord_x[:, None], p_coord_y[:, None]), dim=-1)
        radius = int(self.model_cfg.TARGET_ASSIGNER_CONFIG.P_RADIUS)  # 高斯半径
        heatmap = sy_draw_gaussian_to_heatmap(heatmap, p_center, radius, 1)

        # 生成cyc heatmap 真值
        coord_cy = data_dict['cy_bm_spatial_features'][bs_idx].reshape(496, 432)
        coord_cy = (coord_cy == 1).nonzero(as_tuple=False)
        cy_coord_x = coord_cy[:, 0]
        cy_coord_y = coord_cy[:, 1]
        cy_center = torch.cat((cy_coord_x[:, None], cy_coord_y[:, None]), dim=-1)
        radius = int(self.model_cfg.TARGET_ASSIGNER_CONFIG.CY_RADIUS)  # 高斯半径
        heatmap = sy_draw_gaussian_to_heatmap(heatmap, cy_center, radius, 2)

        # 可视化car bev shape mask
        # vis = data_dict['c_bm_spatial_features'].reshape(496, 432)
        # fig = plt.figure(figsize=(10, 10))
        # sns.heatmap(vis.cpu().numpy())
        # plt.show()
        # vis = heatmap[0]
        # fig = plt.figure(figsize=(10, 10))
        # sns.heatmap(vis.cpu().numpy())
        # plt.show()
        # vis = heatmap[1]
        # fig = plt.figure(figsize=(10, 10))
        # sns.heatmap(vis.cpu().numpy())
        # plt.show()
        # vis = heatmap[2]
        # fig = plt.figure(figsize=(10, 10))
        # sns.heatmap(vis.cpu().numpy())
        # plt.show()
        return heatmap

    def assign_targets(self, gt_boxes, feature_map_size=None, **kwargs):
        """
        在assign_targets这个函数中，利用真值生成了基于高斯分布的heatmap,
        即基于指定的iou阈值生成gaussian radius，然后在范围内生成heatmap,与CenterNet没什么区别。
        Args:
            gt_boxes: (B, M, 8)
            range_image_polar: (B, 3, H, W)
            feature_map_size: (2) [H, W]
            spatial_cartesian: (B, 4, H, W)
        Returns:

        """
        feature_map_size = feature_map_size[::-1]  # [H, W] ==> [x, y]      [176 200] to [200 176]
        target_assigner_cfg = self.model_cfg.TARGET_ASSIGNER_CONFIG
        # feature_map_size = self.grid_size[:2] // target_assigner_cfg.FEATURE_MAP_STRIDE

        batch_size = gt_boxes.shape[0]
        ret_dict = {
            'heatmaps': [],
            'target_boxes': [],
            'inds': [],
            'masks': [],
            'heatmap_masks': []
        }

        all_names = np.array(['bg', *self.class_names])
        for idx, cur_class_names in enumerate(self.class_names_each_head):
            heatmap_list, target_boxes_list, inds_list, masks_list = [], [], [], []
            for bs_idx in range(batch_size):
                cur_gt_boxes = gt_boxes[bs_idx]
                gt_class_names = all_names[cur_gt_boxes[:, -1].cpu().long().numpy()]

                gt_boxes_single_head = []

                for idx, name in enumerate(gt_class_names):
                    if name not in cur_class_names:
                        continue
                    temp_box = cur_gt_boxes[idx]
                    temp_box[-1] = cur_class_names.index(name) + 1
                    gt_boxes_single_head.append(temp_box[None, :])

                if len(gt_boxes_single_head) == 0:
                    gt_boxes_single_head = cur_gt_boxes[:0, :]
                else:
                    gt_boxes_single_head = torch.cat(gt_boxes_single_head, dim=0)

                heatmap, ret_boxes, inds, mask = self.assign_target_of_single_head(
                    num_classes=len(cur_class_names), gt_boxes=gt_boxes_single_head.cpu(),
                    feature_map_size=feature_map_size, feature_map_stride=target_assigner_cfg.FEATURE_MAP_STRIDE,
                    num_max_objs=target_assigner_cfg.NUM_MAX_OBJS,
                    gaussian_overlap=target_assigner_cfg.GAUSSIAN_OVERLAP,
                    min_radius=target_assigner_cfg.MIN_RADIUS,
                )
                heatmap_list.append(heatmap.to(gt_boxes_single_head.device))
                target_boxes_list.append(ret_boxes.to(gt_boxes_single_head.device))
                inds_list.append(inds.to(gt_boxes_single_head.device))
                masks_list.append(mask.to(gt_boxes_single_head.device))

            ret_dict['heatmaps'].append(torch.stack(heatmap_list, dim=0))
            ret_dict['target_boxes'].append(torch.stack(target_boxes_list, dim=0))
            ret_dict['inds'].append(torch.stack(inds_list, dim=0))
            ret_dict['masks'].append(torch.stack(masks_list, dim=0))
        return ret_dict

    def sy_assign_targets(self, data_dict, feature_map_size=None, **kwargs):
        """
        在assign_targets这个函数中，利用真值生成了基于高斯分布的heatmap,
        即基于指定的iou阈值生成gaussian radius，然后在范围内生成heatmap,与CenterNet没什么区别。
        """
        gt_boxes = data_dict['gt_boxes']
        feature_map_size = feature_map_size[::-1]  # [H, W] ==> [x, y]      [176 200] to [200 176]
        batch_size = gt_boxes.shape[0]
        ret_dict = {
            'heatmaps': [],
        }
        for idx, cur_class_names in enumerate(self.class_names_each_head):
            heatmap_list, target_boxes_list, inds_list, masks_list = [], [], [], []
            for bs_idx in range(batch_size):
                heatmap = self.sy_assign_target_of_single_head(
                    num_classes=len(cur_class_names),
                    feature_map_size=feature_map_size,
                    data_dict=data_dict,
                    bs_idx=bs_idx,
                )
                heatmap_list.append(heatmap)
            ret_dict['heatmaps'].append(torch.stack(heatmap_list, dim=0))
        return ret_dict

    def sigmoid(self, x):
        y = torch.clamp(x.sigmoid(), min=1e-4, max=1 - 1e-4)
        return y

    def get_loss_bev_shape(self):
        pred_dicts = self.forward_ret_dict['pred_dicts']
        target_dicts = self.forward_ret_dict['target_dicts']

        # 可视化真值和预测值
        # hm = self.sigmoid(pred_dicts[0]['hm'][0][0])
        # fig = plt.figure(figsize=(10, 10))
        # sns.heatmap(target_dicts['heatmaps'][0][0][0].cpu().numpy())
        # plt.show()
        # fig = plt.figure(figsize=(10, 10))
        # sns.heatmap(hm.detach().cpu().numpy())
        # plt.show()
        # mask = hm > 0.6
        # hm[mask] = 1
        # hm[~mask] = 0
        # fig = plt.figure(figsize=(10, 10))
        # sns.heatmap(hm.detach().cpu().numpy())
        # plt.show()

        tb_dict = {}
        loss = 0

        # 只包含hm loss
        for idx, pred_dict in enumerate(pred_dicts):
            pred_dict['hm'] = self.sigmoid(pred_dict['hm'])
            hm_loss = self.hm_loss_func(pred_dict['hm'], target_dicts['heatmaps'][idx])
            hm_loss *= self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['cls_weight']

            loss += hm_loss
            tb_dict['shape_hm_loss_head_%d' % idx] = hm_loss.item()
        return loss, tb_dict

    def forward(self, data_dict):
        spatial_features_2d = data_dict['spatial_features_2d']  # [bs,384,496,432]
        x = self.shared_conv(spatial_features_2d)  # shared conv是64，这里做了降维得到【bs,64,496,432】

        pred_dicts = []
        for head in self.heads_list:
            pred_dicts.append(head(x))  # 对降维到64后的BEV feature每个像素位置预测center 1、centerz 1、dim 3、rot 2、hm 1

        if data_dict['train_bev_shape'] == True:
            # 利用真值生成了基于高斯分布的heatmap
            target_dict = self.sy_assign_targets(
                data_dict, feature_map_size=spatial_features_2d.size()[2:],
                feature_map_stride=data_dict.get('spatial_features_2d_strides', None)
            )
            self.forward_ret_dict['target_dicts'] = target_dict

        self.forward_ret_dict['pred_dicts'] = pred_dicts  # 每个feature map像素(200*176)预测center、centerz、dim、rot、hm
        data_dict['bev_hm'] = self.sigmoid(pred_dicts[0]['hm'])


        # 按阈值二分类
        hm_binary = copy.deepcopy(data_dict['bev_hm'].detach())
        hm_prob = copy.deepcopy(data_dict['bev_hm'].detach())

        mask = hm_binary > 0.3
        hm_binary[mask] = 1
        hm_binary[~mask] = 0
        hm_prob[~mask] = 0

        data_dict['hm_binary'] = hm_binary
        data_dict['hm_prob'] = hm_prob

        hm_binary_fuse = []
        hm_prob_fuse = []
        for i in range(data_dict['batch_size']):
            hm_binary_tensor = hm_binary[i, 0, :, :]+hm_binary[i, 1, :, :]+hm_binary[i, 2, :, :]
            hm_binary_fuse.append(hm_binary_tensor.unsqueeze(0))

            hm_prob_tensor = hm_prob[i, 0, :, :]+hm_prob[i, 1, :, :]+hm_prob[i, 2, :, :]
            hm_prob_fuse.append(hm_prob_tensor.unsqueeze(0))

        hm_binary_fuse = torch.cat(hm_binary_fuse)
        hm_binary_fuse = torch.clamp(hm_binary_fuse, max=1)
        data_dict['hm_binary_fuse'] = hm_binary_fuse.unsqueeze(1)

        hm_prob_fuse = torch.cat(hm_prob_fuse)
        hm_prob_fuse = torch.clamp(hm_prob_fuse, max=1)
        data_dict['hm_prob_fuse'] = hm_prob_fuse.unsqueeze(1)

        # 可视化融合的二分类黑白mask
        # vis = data_dict['hm_binary_fuse'][0][0].detach()
        # fig = plt.figure(figsize=(10, 10))
        # sns.heatmap(vis.cpu().numpy())
        # plt.show()


        # 可视化bev 热图真值
        # vis = target_dict['heatmaps'][0][0][0]
        # fig = plt.figure(figsize=(10, 10))
        # sns.heatmap(vis.cpu().numpy())
        # plt.show()
        # vis = target_dict['heatmaps'][0][0][1]
        # fig = plt.figure(figsize=(10, 10))
        # sns.heatmap(vis.cpu().numpy())
        # plt.show()
        # vis = target_dict['heatmaps'][0][0][2]
        # fig = plt.figure(figsize=(10, 10))
        # sns.heatmap(vis.cpu().numpy())
        # plt.show()
        return data_dict
