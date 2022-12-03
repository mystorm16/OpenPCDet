import time
import torch
from .detector3d_template import Detector3DTemplate
from tools.visual_utils.open3d_vis_utils import draw_scenes, draw_scenes_voxel_a, draw_scenes_voxel_b, \
    draw_spherical_voxels_index, draw_spherical_voxels_points
import matplotlib.pyplot as plt
import seaborn as sns;
sns.set()

class Bev_Shape_Pillar(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.bev_shape_module_lists, self.module_list = self.build_networks()
        self.train_bev_shape = model_cfg.BEV_SHAPE.TRAIN_BEV_SHAPE

    def forward(self, batch_dict):
        batch_dict['train_bev_shape'] = self.train_bev_shape
        for cur_module in self.bev_shape_module_lists:
            batch_dict = cur_module(batch_dict)

        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}
        loss_bev_shape, tb_dict = self.bev_shape_modules.dense_head.get_loss_bev_shape()
        loss_rpn, tb_dict = self.dense_head.get_loss()
        loss_bev_shape = 10*loss_bev_shape
        loss = loss_rpn + loss_bev_shape

        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            'loss_bev_shape': loss_bev_shape.item(),
            **tb_dict
        }

        return loss, tb_dict, disp_dict
