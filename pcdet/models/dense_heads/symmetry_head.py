# 前景点分割网络构建
import numpy as np
import torch
import tools.visual_utils.open3d_vis_utils as V

# 这里的num_class是3 预测x y z坐标
from pcdet.models.dense_heads.point_head_template import PointHeadTemplate

pdist = torch.nn.PairwiseDistance(p=2)  # 计算欧式距离


class SymmetryHead(PointHeadTemplate):
    def __init__(self, num_class, input_channels, model_cfg, **kwargs):
        super().__init__(model_cfg=model_cfg,num_class=num_class)
        # 根据yaml构建网络
        self.symmetry_layers = self.make_fc_layers(
            fc_cfg=self.model_cfg.CLS_FC,
            input_channels=input_channels,
            output_channels=2
        )

    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict
        symmetry_preds = self.forward_ret_dict['point_symmetry_preds']
        symmetry_labels_csc = self.forward_ret_dict['symmetry_one_frame_car_csc']
        symmetry_loss = pdist(symmetry_preds, symmetry_labels_csc)
        symmetry_loss = torch.sum(symmetry_loss)/symmetry_loss.shape[0]
        if tb_dict is None:
            tb_dict = {}
        tb_dict.update({
            'point_loss_cls': symmetry_loss.item(),
        })
        return symmetry_loss, tb_dict

    def forward(self, batch_dict):
        if self.model_cfg.get('USE_POINT_FEATURES_BEFORE_FUSION', False):
            point_features = batch_dict['point_features_before_fusion']
        else:
            point_features = batch_dict['point_features']
        point_symmetry_preds = self.symmetry_layers(point_features)  # 估计的前景点对称点

        if self.training ==True:
            ret_dict = {
                'point_symmetry_preds':point_symmetry_preds,
                # 'symmetry_one_frame_car':batch_dict['symmetry_in_frame'][:,1:3],
                'symmetry_one_frame_car_csc':batch_dict['symmetry_in_frame_csc'][:,1:3]
            }
            self.forward_ret_dict = ret_dict
        else:
            idx = batch_dict['symmetry_idx']
            point_symmetry_preds = point_symmetry_preds + batch_dict['pred_dicts'][0]['pred_boxes'][idx][:2]
            point_symmetry_preds = torch.cat((point_symmetry_preds, batch_dict['point_coords'][:, 3].unsqueeze(1)), dim=1)
            batch_dict.update(point_symmetry_preds=point_symmetry_preds)
            point_symmetry_preds =  torch.nn.functional.pad(point_symmetry_preds, (1, 1), 'constant', 0)  # 补齐反射强度
            point_symmetry_preds_fuse =  torch.cat((batch_dict['points'],point_symmetry_preds), dim=0)
            batch_dict.update(point_symmetry_preds_fuse=point_symmetry_preds_fuse)
            V.draw_scenes(
                points=point_symmetry_preds_fuse[:,1:],
                gt_boxes=batch_dict['gt_boxes'][0],
                ref_boxes=batch_dict['pred_dicts'][0]['pred_boxes'],
                draw_origin=True
            )
        # torch.save(batch_dict,'/media/mystorm/T7/OpenPCDet/per_sense/points9_iou0.45_frame000006_symmetry2.pth')

        return batch_dict





