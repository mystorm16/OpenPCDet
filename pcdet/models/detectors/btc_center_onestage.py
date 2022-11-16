from tools.visual_utils.open3d_vis_utils import draw_scenes, draw_scenes_voxel_a, draw_scenes_voxel_b, \
    draw_spherical_voxels_index, draw_spherical_voxels_points
# from tools.visual_utils.visualize_utils import draw_spherical_voxels, draw_scenes
from .detector3d_template import Detector3DTemplate
import torch
import numpy as np

EVL_VIS = 800
class Btc_Center_Onestage(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset, full_config=None):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset, full_config=full_config)
        self.occ_module_list, self.center_module_list = self.build_networks()
        self.percentage = model_cfg.OCC.PARAMS.PERCENTAGE

    def forward(self, batch_dict):
        # draw_scenes(batch_dict['occ_voxels'].reshape(-1, 4))
        # draw_scenes_voxel_a(batch_dict['occ_voxel_coords'])
        # draw_scenes(batch_dict['det_voxels'].reshape(-1, 4))
        # draw_scenes_voxel_a(batch_dict['det_voxel_coords'])

        use_occ_prob = [True for i in range(batch_dict["batch_size"])]  # False
        prob = np.random.uniform(size=batch_dict["batch_size"], high=0.9999)
        if batch_dict["is_train"]:
            use_occ_prob = prob <= self.percentage
        batch_dict["use_occ_prob"] = use_occ_prob

        for cur_module in self.center_module_list:
            batch_dict = cur_module(batch_dict)
        if batch_dict.__contains__('center_area'):  # 没检测到center
            for cur_module in self.occ_module_list:
                batch_dict = cur_module(batch_dict)

        if not batch_dict["is_train"]:
            self.eval_count += 1

        if self.training:
            loss, det_tb_dict, disp_dict = self.get_training_loss(batch_dict)
            ret_dict = {
                'loss': loss
            }
            return ret_dict, det_tb_dict, disp_dict
        else:
            metric_dicts = {}
            if hasattr(self.model_cfg, "OCC") and hasattr(self.model_cfg.OCC, 'OCC_POST_PROCESSING'):
                occ_dicts, batch_dict = self.occ_post_processing(batch_dict)
                metric_dicts.update(occ_dicts)
            if hasattr(self.model_cfg, 'POST_PROCESSING'):
                pred_dicts, recall_dicts = self.post_processing(batch_dict)
                if self.model_cfg.get('OCC',
                                      None) is not None and self.eval_count % self.model_cfg.OCC.OCC_PNT_UPDATE.VIS.STEP_STRIDE == 0:
                    pc_dict.update(pred_dicts[bind])
                elif self.model_cfg.get('OCC', None) is None and self.eval_count % EVL_VIS == 0:
                    gt_points = self.filter_by_bind(batch_dict["points"][..., 0], bind, batch_dict["points"][..., 1:4])
                    pc_dict.update(pred_dicts[bind])
                    pc_dict.update({
                        "gt_points": gt_points,
                        "gt_boxes": batch_dict["gt_boxes"][bind, :batch_dict["gt_boxes_num"][bind], ...]
                    })
                metric_dicts.update(recall_dicts)
                # draw_scenes(batch_dict['points'][:, 1:], gt_boxes=batch_dict['gt_boxes'][0],
                #             ref_boxes=pred_dicts[0]['pred_boxes'])
                # draw_scenes(batch_dict['voxel_features'], gt_boxes=batch_dict['gt_boxes'][0],
                #             ref_boxes=pred_dicts[0]['pred_boxes'])
                return pred_dicts, metric_dicts, tb_dict, pc_dict
            else:
                return {'loss': torch.zeros(44)}, metric_dicts, {}, pc_dict

    def get_training_loss(self, batch_dict):
        disp_dict = {}
        tb_dict = {}
        occ_loss_rpn, loss_rpn = 0, 0
        if batch_dict.__contains__('center_area'):  # 没检测到center
            occ_loss_rpn, occ_tb_scalar_dict = self.occ_modules.occ_dense_head.get_loss(batch_dict)
        loss_rpn, tb_dict = self.center_modules.dense_head.get_loss_center()

        loss = occ_loss_rpn+loss_rpn

        tb_dict['loss_rpn'] = loss_rpn
        tb_dict['occ_loss_rpn'] = occ_loss_rpn

        return loss, tb_dict, disp_dict
