import copy
import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_
from ..model_utils import model_nms_utils
from ..model_utils import centernet_utils
from ...utils import loss_utils
import seaborn as sns;

sns.set()
import matplotlib.pyplot as plt
import tools.visual_utils.open3d_vis_utils as V


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


class CenterHead(nn.Module):
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
        self.add_module('reg_loss_func', loss_utils.RegLossCenterNet())

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

    def sigmoid(self, x):
        y = torch.clamp(x.sigmoid(), min=1e-4, max=1 - 1e-4)
        return y

    def get_loss_center(self):
        pred_dicts = self.forward_ret_dict['pred_dicts']
        target_dicts = self.forward_ret_dict['target_dicts']

        tb_dict = {}
        loss = 0

        # center head分为三部分loss=热图hm loss + box回归loss
        for idx, pred_dict in enumerate(pred_dicts):
            pred_dict['hm'] = self.sigmoid(pred_dict['hm'])
            hm_loss = self.hm_loss_func(pred_dict['hm'], target_dicts['heatmaps'][idx])
            hm_loss *= self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['cls_weight']

            target_boxes_centers = target_dicts['target_boxes'][idx][:, :, 0:3]  # 1 500 2
            pred_centers = pred_dict['center']  # 1 2 400 352
            pred_center_z = pred_dict['center_z']  # 1 1 400 352
            p_cat = torch.cat((pred_centers,pred_center_z),dim=1)

            # center回归loss
            reg_loss = self.reg_loss_func(
                p_cat, target_dicts['masks'][idx], target_dicts['inds'][idx], target_boxes_centers
            )
            loc_loss = (reg_loss * reg_loss.new_tensor(self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['code_weights'])).sum()
            loc_loss = loc_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']

            loss += hm_loss + loc_loss
            tb_dict['center_area_hm_loss_head_%d' % idx] = hm_loss.item()
            tb_dict['center_area_loc_loss_head_%d' % idx] = loc_loss.item()

        tb_dict['center_area_rpn_loss'] = loss.item()
        return loss, tb_dict

    def get_loss(self):
        pred_dicts = self.forward_ret_dict['pred_dicts']
        target_dicts = self.forward_ret_dict['target_dicts']

        tb_dict = {}
        loss = 0

        # center head分为三部分loss=热图hm loss + box回归loss
        for idx, pred_dict in enumerate(pred_dicts):
            pred_dict['hm'] = self.sigmoid(pred_dict['hm'])
            hm_loss = self.hm_loss_func(pred_dict['hm'], target_dicts['heatmaps'][idx])
            hm_loss *= self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['cls_weight']

            target_boxes = target_dicts['target_boxes'][idx]
            pred_boxes = torch.cat([pred_dict[head_name] for head_name in self.separate_head_cfg.HEAD_ORDER], dim=1)

            reg_loss = self.reg_loss_func(
                pred_boxes, target_dicts['masks'][idx], target_dicts['inds'][idx], target_boxes
            )  # box回归loss
            loc_loss = (reg_loss * reg_loss.new_tensor(self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['code_weights'])).sum()
            loc_loss = loc_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']

            loss += hm_loss + loc_loss
            tb_dict['center_det_hm_loss_head_%d' % idx] = hm_loss.item()
            tb_dict['center_det_loc_loss_head_%d' % idx] = loc_loss.item()

        tb_dict['center_det_rpn_loss'] = loss.item()
        return loss, tb_dict

    def generate_predicted_boxes(self, batch_size, pred_dicts):
        post_process_cfg = self.model_cfg.POST_PROCESSING  # 读取参数
        post_center_limit_range = torch.tensor(post_process_cfg.POST_CENTER_LIMIT_RANGE).cuda().float()  # 中心点范围

        ret_dict = [{
            'pred_boxes': [],
            'pred_scores': [],
            'pred_labels': [],
        } for k in range(batch_size)]
        for idx, pred_dict in enumerate(pred_dicts):
            # 这里主要是遍历每个预测结果 保存一下
            batch_hm = pred_dict['hm'].sigmoid()  # （预测值）过于发散的数据不好学习，用sigmoid稳定一下
            batch_center = pred_dict['center']
            batch_center_z = pred_dict['center_z']
            batch_dim = pred_dict['dim'].exp()
            batch_rot_cos = pred_dict['rot'][:, 0].unsqueeze(dim=1)
            batch_rot_sin = pred_dict['rot'][:, 1].unsqueeze(dim=1)
            batch_vel = pred_dict['vel'] if 'vel' in self.separate_head_cfg.HEAD_ORDER else None

            # 根据hm选取pred box，先找hm值最大的500个位置，再用阈值筛一遍，筛完还有50个左右的box
            # ①先找hm最大的k个位置
            # ②场景范围过滤+hm得分过滤
            final_pred_dicts = centernet_utils.decode_bbox_from_heatmap(
                heatmap=batch_hm, rot_cos=batch_rot_cos, rot_sin=batch_rot_sin,
                center=batch_center, center_z=batch_center_z, dim=batch_dim, vel=batch_vel,
                point_cloud_range=self.point_cloud_range, voxel_size=self.voxel_size,
                feature_map_stride=self.feature_map_stride,
                K=post_process_cfg.MAX_OBJ_PER_SAMPLE,
                circle_nms=(post_process_cfg.NMS_CONFIG.NMS_TYPE == 'circle_nms'),
                score_thresh=post_process_cfg.SCORE_THRESH,
                post_center_limit_range=post_center_limit_range
            )

            for k, final_dict in enumerate(final_pred_dicts):  # 遍历每个batch
                final_dict['pred_labels'] = self.class_id_mapping_each_head[idx][final_dict['pred_labels'].long()]
                if post_process_cfg.NMS_CONFIG.NMS_TYPE != 'circle_nms':
                    selected, selected_scores = model_nms_utils.class_agnostic_nms(
                        box_scores=final_dict['pred_scores'], box_preds=final_dict['pred_boxes'],
                        nms_config=post_process_cfg.NMS_CONFIG,
                        score_thresh=None
                    )  # 用score nms再筛一次，这里用的

                    final_dict['pred_boxes'] = final_dict['pred_boxes'][selected]
                    final_dict['pred_scores'] = selected_scores
                    final_dict['pred_labels'] = final_dict['pred_labels'][selected]

                ret_dict[k]['pred_boxes'].append(final_dict['pred_boxes'])
                ret_dict[k]['pred_scores'].append(final_dict['pred_scores'])
                ret_dict[k]['pred_labels'].append(final_dict['pred_labels'])

        for k in range(batch_size):
            ret_dict[k]['pred_boxes'] = torch.cat(ret_dict[k]['pred_boxes'], dim=0)
            ret_dict[k]['pred_scores'] = torch.cat(ret_dict[k]['pred_scores'], dim=0)
            ret_dict[k]['pred_labels'] = torch.cat(ret_dict[k]['pred_labels'], dim=0) + 1

        return ret_dict

    def generate_predicted_centers(self, batch_size, pred_dicts):
        post_process_cfg = self.model_cfg.POST_PROCESSING  # 读取参数
        post_center_limit_range = torch.tensor(post_process_cfg.POST_CENTER_LIMIT_RANGE).cuda().float()  # 中心点范围

        ret_dict = [{
            'pred_centers': [],
            'pred_scores': [],
        } for k in range(batch_size)]
        for idx, pred_dict in enumerate(pred_dicts):
            batch_hm = pred_dict['hm'].sigmoid()  # （预测值）过于发散的数据不好学习，用sigmoid稳定一下
            batch_center = pred_dict['center']
            batch_center_z = pred_dict['center_z']

            final_pred_center_dicts = centernet_utils.decode_centers_from_heatmap(
                heatmap=batch_hm, center=batch_center, center_z=batch_center_z, point_cloud_range=self.point_cloud_range,
                voxel_size=self.voxel_size, feature_map_stride=self.feature_map_stride,
                K=post_process_cfg.MAX_OBJ_PER_SAMPLE,
                circle_nms=(post_process_cfg.NMS_CONFIG.NMS_TYPE == 'circle_nms'),
                score_thresh=post_process_cfg.SCORE_THRESH,
                post_center_limit_range=post_center_limit_range,
                min_radius=post_process_cfg.NMS_CONFIG.MIN_RADIUS,
                post_max_size=post_process_cfg.NMS_CONFIG.POST_MAX_SIZE,
            )

            for k, final_dict in enumerate(final_pred_center_dicts):
                ret_dict[k]['pred_centers'].append(final_dict['pred_centers'])
                ret_dict[k]['pred_scores'].append(final_dict['pred_scores'])

        for k in range(batch_size):
            ret_dict[k]['pred_centers'] = torch.cat(ret_dict[k]['pred_centers'], dim=0)
            ret_dict[k]['pred_scores'] = torch.cat(ret_dict[k]['pred_scores'], dim=0)

        return ret_dict



    @staticmethod
    def reorder_rois_for_refining(batch_size, pred_dicts):
        num_max_rois = max([len(cur_dict['pred_boxes']) for cur_dict in pred_dicts])
        num_max_rois = max(1, num_max_rois)  # at least one faked rois to avoid error
        pred_boxes = pred_dicts[0]['pred_boxes']

        rois = pred_boxes.new_zeros((batch_size, num_max_rois, pred_boxes.shape[-1]))
        roi_scores = pred_boxes.new_zeros((batch_size, num_max_rois))
        roi_labels = pred_boxes.new_zeros((batch_size, num_max_rois)).long()

        for bs_idx in range(batch_size):
            num_boxes = len(pred_dicts[bs_idx]['pred_boxes'])

            rois[bs_idx, :num_boxes, :] = pred_dicts[bs_idx]['pred_boxes']
            roi_scores[bs_idx, :num_boxes] = pred_dicts[bs_idx]['pred_scores']
            roi_labels[bs_idx, :num_boxes] = pred_dicts[bs_idx]['pred_labels']
        return rois, roi_scores, roi_labels

    def forward(self, data_dict):
        spatial_features_2d = data_dict['spatial_features_2d']  # BEV Feature [bs,2*256,200,176]
        x = self.shared_conv(spatial_features_2d)  # shared conv是64，这里做了降维得到【bs,64,200,176】

        pred_dicts = []
        for head in self.heads_list:
            pred_dicts.append(head(x))  # 对降维到64后的BEV feature每个像素位置预测center 1、centerz 1、dim 3、rot 2、hm 1

        if self.training:
            # assign_targets中，利用真值生成了基于高斯分布的heatmap
            target_dict = self.assign_targets(
                data_dict['gt_boxes'], feature_map_size=spatial_features_2d.size()[2:],
                feature_map_stride=data_dict.get('spatial_features_2d_strides', None)
            )
            self.forward_ret_dict['target_dicts'] = target_dict

        self.forward_ret_dict['pred_dicts'] = pred_dicts  # 每个feature map像素(200*176)预测center、centerz、dim、rot、hm

        if not self.training or self.predict_boxes_when_training:  # 在预测center、centerz、dim、rot、hm后预测box 阈值+NMS
            pred_dicts = self.generate_predicted_boxes(  # 在训练过程中预测box、cls score、label
                data_dict['batch_size'], pred_dicts
            )

            if self.predict_boxes_when_training:
                rois, roi_scores, roi_labels = self.reorder_rois_for_refining(data_dict['batch_size'],
                                                                              pred_dicts)  # 第一阶段检测结果
                data_dict['rois'] = rois
                data_dict['roi_scores'] = roi_scores
                data_dict['roi_labels'] = roi_labels
                data_dict['has_class_labels'] = True
            else:
                data_dict['final_box_dicts'] = pred_dicts

        # center区域生成
        if self.model_cfg.POST_PROCESSING.GENERATE_CENTERS:  # test 时候使用 batch=1
            pred_center_dicts = self.generate_predicted_centers(
                data_dict['batch_size'], pred_dicts
            )

            batch_size = data_dict['batch_size']
            for bs in range(batch_size):
                if pred_center_dicts[bs]['pred_centers'].shape[0] == 0:  # 没检测到center
                    return data_dict
            data_dict['final_centers_dicts'] = pred_center_dicts  # center存入batch dict
            pred_centers_list = []
            for bs_idx in range(batch_size):
                pred_centers = torch.nn.functional.pad(pred_center_dicts[bs_idx]['pred_centers'],
                                                          (1, 0), 'constant', bs_idx)  # 加入bs_idx
                pred_centers_list.append(pred_centers)
            pred_centers = torch.cat(pred_centers_list, dim=0)
            data_dict['final_centers'] = pred_centers  # center存入batch dict

            # center_area = []
            # for bs_idx in range(batch_size):
            #     cur_center_area = []
            #     center_R = self.model_cfg.POST_PROCESSING.CENTERS_RADIUS  # 区域半径
            #     center_density = self.model_cfg.POST_PROCESSING.CENTERS_DENSITY  # 区域密度
            #     hm_center = pred_center_dicts[bs_idx]['pred_centers'][:, :2]
            #     for k, y in enumerate(hm_center):  # y是中心
            #         for i in range(-center_R, center_R+1):  # 这里代表搜索区域。i，j表示相对于center的偏移量
            #             for j in range(-center_R, center_R+1):
            #                 if self.model_cfg.POST_PROCESSING.AREA_TYPE == 'Euclidean':
            #                     if np.linalg.norm([i, j]) <= center_R:  # 欧式距离
            #                         for n in range(-3, 2):
            #                             cur_center_area.append([y[0] + i/center_density, y[1] + j/center_density, n/2])
            #                 elif self.model_cfg.POST_PROCESSING.AREA_TYPE == 'Manhattan':
            #                     if abs(i) + abs(j) <= center_R:  # 曼哈顿距离
            #                         for n in range(-3, 2):
            #                             cur_center_area.append([y[0] + i/center_density, y[1] + j/center_density, n/2])
            #                 else:  # Rectangle
            #                     for n in range(-3, 2):
            #                         cur_center_area.append([y[0] + i/center_density, y[1] + j/center_density, n/2])
            #     cur_center_area = torch.tensor(cur_center_area).float().cuda()
            #     cur_center_area = torch.nn.functional.pad(cur_center_area, (1, 0), 'constant', bs_idx)  # 加入bs_idx
            #     center_area.append(cur_center_area)
            #     '''可视化center area'''
            #     # points = torch.vstack((data_dict['points'][:, 1:4], cur_center_area[:, 1:]))
            #     # V.draw_scenes(points, gt_boxes=data_dict['gt_boxes'][bs_idx], draw_origin=True)
            #     # points = cur_center_area
            #     # V.draw_scenes(points[:, 1:], gt_boxes=data_dict['gt_boxes'][bs_idx], draw_origin=True)
            #
            # # 没检测到center
            # for bs in range(batch_size):
            #     if center_area[bs].shape[0] == 1:
            #         return data_dict
            #
            # center_area = torch.cat(center_area, dim=0)
            # data_dict['center_area'] = center_area
            # # V.draw_scenes(center_area[:, 1:])

        return data_dict
