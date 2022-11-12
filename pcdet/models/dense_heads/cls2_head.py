# 前景点分割网络构建
import torch

from .cls2_head_template import CLS2_HeadTemplate
from ...utils import box_coder_utils, box_utils


class ClsHead(CLS2_HeadTemplate):  # 从PointHeadTemplate调用make_fc_layers
    def __init__(self, num_class, input_channels, model_cfg, **kwargs):
        super().__init__(model_cfg=model_cfg, num_class=num_class)
        # 根据yaml构建网络
        self.cls_layers = self.make_fc_layers(
            fc_cfg=self.model_cfg.CLS_FC,
            input_channels=input_channels,
            output_channels=num_class
        )

    # 预测的前景点和前景点真值的关系
    def assign_targets(self, input_dict):
        """
        Args:
            input_dict:
                point_features: (N1 + N2 + N3 + ..., C)
                batch_size:
                point_coords: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
                gt_boxes (optional): (B, M, 8)
        Returns:
            point_cls_labels: (N1 + N2 + N3 + ...), long type, 0:background, -1:ignored
            point_part_labels: (N1 + N2 + N3 + ..., 3)
        """
        # 得到每个点的坐标 shape（bacth * 16384, 4），其中4个维度分别是batch_id，x，y，z
        point_coords = input_dict['point_coords']
        # 取出gt_box，shape （batch， num_of_GTs, 8），
        # 其中维度8表示 x, y, z, l, w, h, heading, class_id
        gt_boxes = input_dict['gt_boxes']
        assert gt_boxes.shape.__len__() == 3, 'gt_boxes.shape=%s' % str(gt_boxes.shape)
        assert point_coords.shape.__len__() in [2], 'points.shape=%s' % str(point_coords.shape)

        batch_size = gt_boxes.shape[0]
        extend_gt_boxes = box_utils.enlarge_box3d(
            gt_boxes.view(-1, gt_boxes.shape[-1]), extra_width=self.model_cfg.TARGET_CONFIG.GT_EXTRA_WIDTH
        ).view(batch_size, -1, gt_boxes.shape[-1])
        """
        assign_stack_targets函数完成了一批数据中所有点的前背景分配，
        并为每个前景点分配了对应的类别和box的7个回归参数，xyzlwhθ
        """
        targets_dict = self.assign_stack_targets(
            points=point_coords, gt_boxes=gt_boxes, extend_gt_boxes=extend_gt_boxes,
            set_ignore_flag=True, use_ball_constraint=False,
            ret_part_labels=False, ret_box_labels=True
        )

        return targets_dict

    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict
        point_loss_cls, tb_dict_1 = self.get_cls_layer_loss()

        point_loss = point_loss_cls
        tb_dict.update(tb_dict_1)
        return point_loss, tb_dict

    def forward(self, batch_dict):
        point_features = batch_dict['point_features']  # 从字典中获取每个点的特征 shape （batch * 16384, 128）
        point_cls_preds = self.cls_layers(point_features)  # (total_points, num_class)
        # 从每个点的分类预测结果中，取出类别预测概率最大的结果  (batch * 16384, num_class) --> (batch * 16384, )
        point_cls_preds_max, _ = point_cls_preds.max(dim=-1)
        # 将类别预测分数经过sigmoid激活后放入字典中
        batch_dict['point_cls_scores'] = torch.sigmoid(point_cls_preds_max)
        # 将点的类别预测结果和回归结果放入字典中
        ret_dict = {
            'point_cls_preds': point_cls_preds,
        }
        # 如果在训练模式下，需要根据GTBox来生成对应的前背景点，用于点云的前背景分割，给后面计算前背景分类loss
        if self.training:
            targets_dict = self.assign_targets(batch_dict)
            # 将一个batch中所有点的GT类别结果放入字典中 shape (batch * 16384)
            ret_dict['point_cls_labels'] = targets_dict['point_cls_labels']
        # 第一阶段生成的预测结果放入前向传播字典
        self.forward_ret_dict = ret_dict

        point_cls_labels = targets_dict['point_cls_labels']

        # 分析label中的类别数量
        outputf1 = torch.nonzero(point_cls_labels == -1).squeeze().size()
        outputl0 = torch.nonzero(point_cls_labels == 0).squeeze().size()
        outputl1 = torch.nonzero(point_cls_labels == 1).squeeze().size()
        outputl2 = torch.nonzero(point_cls_labels == 2).squeeze().size()
        outputl3 = torch.nonzero(point_cls_labels == 3).squeeze().size()

        # 分析predict中的类别数量
        outputp = point_cls_preds.argmax(dim=1)
        outputpuni = torch.unique(outputp)
        outputp0 = torch.nonzero(outputp == 0).squeeze().size()
        outputp1 = torch.nonzero(outputp == 1).squeeze().size()
        outputp2 = torch.nonzero(outputp == 2).squeeze().size()

        return batch_dict
