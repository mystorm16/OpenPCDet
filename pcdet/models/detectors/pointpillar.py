from .detector3d_template import Detector3DTemplate
from tools.visual_utils.open3d_vis_utils import draw_scenes, draw_scenes_voxel_a, draw_scenes_voxel_b, \
    draw_spherical_voxels_index, draw_spherical_voxels_points
import time

class PointPillar(Detector3DTemplate):
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
            t1 = time.perf_counter()
            loss, tb_dict, disp_dict = self.get_training_loss()
            t2 = time.perf_counter()
            print(t2 - t1)

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}

        loss_rpn, tb_dict = self.dense_head.get_loss()
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            **tb_dict
        }

        loss = loss_rpn
        return loss, tb_dict, disp_dict
