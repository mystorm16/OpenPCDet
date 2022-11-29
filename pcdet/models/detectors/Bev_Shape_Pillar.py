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
        _, _, _, self.module_list = self.build_networks()

    def forward(self, batch_dict):
        for cur_module in self.module_list:
            t1 = time.perf_counter()
            batch_dict = cur_module(batch_dict)
            t2 = time.perf_counter()
            print(t2-t1)

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

        loss_bev_shape, tb_dict = self.dense_head.get_loss_bev_shape()
        tb_dict = {
            'loss_bev_shape': loss_bev_shape.item(),
            **tb_dict
        }

        loss = loss_bev_shape
        return loss, tb_dict, disp_dict

    def post_processing(self, batch_dict):
        score_thresh = self.model_cfg.POST_PROCESSING.SCORE_THRESH

        # 按阈值二分类
        hm_binary = batch_dict['bev_hm'].sigmoid()
        mask = hm_binary > score_thresh
        hm_binary[mask] = 1
        hm_binary[~mask] = 0
        batch_dict['hm_binary'] = hm_binary

        # vis = batch_dict['hm_binary'][0][0]
        # fig = plt.figure(figsize=(10, 10))
        # sns.heatmap(vis.cpu().numpy())
        # plt.show()
        return batch_dict