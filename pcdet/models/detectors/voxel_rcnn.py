import torch

from tools.visual_utils.open3d_vis_utils import draw_scenes_voxel_a, draw_scenes

from .detector3d_template import Detector3DTemplate


class VoxelRCNN(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        _, _, _, self.module_list = self.build_networks()

    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            # 输入 第二阶段优化之后的100个框
            # 输出 最终nms后的结果
            pred_dicts, recall_dicts = self.post_processing(batch_dict)

            # sy add
            # iou_list, points_num_list = iou_ptsnum(batch_dict, pred_dicts)
            # recall_dicts.update(iou=iou_list, points_num=points_num_list)

            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}
        loss = 0
        
        loss_rpn, tb_dict = self.dense_head.get_loss()
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)

        loss = loss + loss_rpn + loss_rcnn
        
        if hasattr(self.backbone_3d, 'get_loss'):
            loss_backbone3d, tb_dict = self.backbone_3d.get_loss(tb_dict)
            loss += loss_backbone3d
            
        return loss, tb_dict, disp_dict
