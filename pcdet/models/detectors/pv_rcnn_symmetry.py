# pvrcnn symmetry的detector
import torch

from .detector3d_template import Detector3DTemplate
from tools.visual_utils import open3d_vis_utils as V


class PVRCNN_symmetry(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        # 在框架中增加额外模块 symmetry_head
        self.module_topology = [
            'vfe', 'backbone_3d', 'map_to_bev_module', 'pfe',
            'backbone_2d', 'dense_head', 'point_head', 'roi_head',
        ]
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        for cur_module in self.module_list:  # 循环model list进行网络预测
            batch_dict = cur_module(batch_dict)
        # torch.save(self.module_list, '/media/mystorm/T7/OpenPCDet/ckpt/symmetry_net.pth')
        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()
            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.cls_post_processing(batch_dict)
            return pred_dicts

    def get_training_loss(self):    # get_training_loss函数 方便不同loss相加
        disp_dict = {}
        loss_point, tb_dict = self.point_head.get_loss()  # 计算一帧的loss

        loss = loss_point
        return loss, tb_dict, disp_dict
