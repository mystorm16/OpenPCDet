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
        self.occ_module_list, self.center_module_list, self.center_det_module_list, _ = self.build_networks()
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
        if batch_dict.__contains__('center_area') or batch_dict.__contains__('final_centers'):  # 没检测到center
            for cur_module in self.occ_module_list:
                batch_dict = cur_module(batch_dict)
            for cur_module in self.center_det_module_list:
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
                metric_dicts.update(recall_dicts)
                # draw_scenes(batch_dict['points'][:, 1:], gt_boxes=batch_dict['gt_boxes'][0],
                #             ref_boxes=pred_dicts[0]['pred_boxes'])
                # draw_scenes(batch_dict['voxel_features'], gt_boxes=batch_dict['gt_boxes'][0],
                #             ref_boxes=pred_dicts[0]['pred_boxes'])
                return pred_dicts, metric_dicts

    def get_training_loss(self, batch_dict):
        disp_dict = {}
        occ_loss_rpn = torch.tensor([0]).cuda()
        loss_center_det = torch.tensor([0]).cuda()

        """occ loss + center det loss"""
        if batch_dict.__contains__('center_area') or batch_dict.__contains__('final_centers'):  # 检测到了center
            occ_loss_rpn, occ_tb_scalar_dict = self.occ_modules.occ_dense_head.get_loss(batch_dict)
            occ_loss_rpn = occ_loss_rpn * 10  # occ loss放大10倍
            loss_center_det, tb_center_det_dict = self.center_det_modules.dense_head.get_loss()

        """center area loss"""
        loss_center_area, tb_dict = self.center_modules.dense_head.get_loss_center()


        loss = loss_center_area + occ_loss_rpn + loss_center_det

        tb_dict['loss_center_area'] = loss_center_area
        tb_dict['loss_center_det'] = loss_center_det
        tb_dict['occ_loss_rpn'] = occ_loss_rpn

        return loss, tb_dict, disp_dict

    def post_processing(self, batch_dict):
        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = batch_dict['batch_size']
        final_pred_dict = batch_dict['final_box_dicts']
        recall_dict = {}
        for index in range(batch_size):
            pred_boxes = final_pred_dict[index]['pred_boxes']

            recall_dict = self.generate_recall_record(
                box_preds=pred_boxes,
                recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                thresh_list=post_process_cfg.RECALL_THRESH_LIST
            )

        return final_pred_dict, recall_dict
