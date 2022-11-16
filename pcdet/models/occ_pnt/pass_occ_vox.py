import torch
from tools.visual_utils.open3d_vis_utils import draw_scenes_voxel_b, draw_scenes_voxel_a, draw_scenes

from .add_occ_template import AddOccTemplate


class PassOccVox(AddOccTemplate):
    def __init__(self, model_cfg, data_cfg, point_cloud_range, occ_voxel_size, occ_grid_size, center_voxel_size,
                 center_grid_size, mode, voxel_centers, **kwargs):
        super().__init__(model_cfg, data_cfg, point_cloud_range, occ_voxel_size, occ_grid_size, center_voxel_size,
                         center_grid_size, mode, voxel_centers)

    def forward(self, batch_dict, **kwargs):
        voxel_features, voxel_num_points, coords = batch_dict['occ_voxels'], batch_dict['occ_voxel_num_points'], \
                                                   batch_dict['occ_voxel_coords']
        _, _, pnt_feat_dim = list(voxel_features.size())

        batch_size, pre_occ_probs = batch_dict['batch_size'], batch_dict['batch_pred_occ_prob']  # 每个体素位置预测的占有概率
        res_lst, probs_lst, occ_coords_lst = self.filter_occ_points(batch_size, pre_occ_probs,
                                                                    batch_dict)  # 滤出占有概率大于阈值的体素 2048个

        # draw_spherical_voxels_index(occ_coords_lst[0][:, 1:])  # 可视化预测出的占有体素
        # draw_spherical_voxels_points(res_lst[0])
        batch_dict["added_occ_xyz"] = None
        batch_dict["occ_pnts"] = None
        batch_dict["added_occ_b_ind"] = None
        batch_dict["gt_points_xyz"] = batch_dict["points"][..., 1:4]
        batch_dict["gt_b_ind"] = batch_dict["points"][..., 0]

        if 'center_voxel_coords' in batch_dict:
            batch_dict['voxels'], batch_dict['voxel_num_points'], batch_dict['voxel_coords'] = \
                batch_dict['center_voxels'], batch_dict['center_voxel_num_points'], batch_dict['center_voxel_coords']

        if len(probs_lst) > 0:  # 前2048个预测结果
            occ_probs = probs_lst[0] if len(probs_lst) == 1 else torch.cat(probs_lst, axis=0)  # 大于占有概率阈值的体素
            occ_coords = occ_coords_lst[0] if len(occ_coords_lst) == 1 else torch.cat(occ_coords_lst, axis=0)  # 这些体素对应的occ序号
            # draw_scenes_voxel_a(occ_coords)
            batch_dict["added_occ_xyz"] = self.occ_coords2absxyz(occ_coords,  # 从occ系index转点云坐标
                                                                 rot_z=batch_dict[
                                                                     "rot_z"] if "rot_z" in batch_dict else None)  # 预测出的大于占有概率阈值的体素的直角坐标
            # draw_scenes(batch_dict["added_occ_xyz"])
            if self.reg:
                occ_res = res_lst[0] if len(res_lst) == 1 else torch.cat(res_lst, axis=0)
                batch_dict["added_occ_xyz"] += occ_res  # 终于知道这个reg咋用的了，是对最终预测出的占有体素进行微调
            occ_pnts = torch.cat([batch_dict["added_occ_xyz"], occ_probs.unsqueeze(-1)], dim=-1)  # 预测的补全上的点的坐标 直角系
            batch_dict["added_occ_b_ind"] = occ_coords[..., 0]  # batch index
            batch_dict["occ_pnts"] = occ_pnts
            # draw_scenes(batch_dict['added_occ_xyz'].detach())

            occ_carte_coords = self.trans_voxel_grid(batch_dict["added_occ_xyz"], occ_coords[..., 0],
                                                     self.center_voxel_size, self.center_grid_size,
                                                     self.point_cloud_range)  # 从原始点云生成COORD
            # draw_scenes_voxel_a(occ_carte_coords)
            occ_pnts = self.assemble_occ_points(batch_dict["added_occ_xyz"], pnt_feat_dim, occ_probs)  # 补全上的点的坐标拼接占有概率
            if self.db_proj:
                occ_pnts, occ_carte_coords = self.db_proj_func(occ_pnts, occ_coords, occ_carte_coords, batch_dict,
                                                               expand=[1.0, 5.0, 3.0], stride=[1.0, 2.5, 1.5])
            gt_points, gt_voxel_coords = self.assemble_gt_vox_points(batch_dict)
            voxels, voxel_num_points, voxel_coords = self.combine_gt_occ_voxel_point(gt_points, gt_voxel_coords,
                                                                                     occ_pnts, occ_carte_coords,
                                                                                     self.center_grid_size)  # 补全的点拼接到原始点云
            # draw_scenes(voxels.reshape(-1, 6).detach(), gt_boxes=batch_dict['gt_boxes'][0])
            batch_dict['voxels'], batch_dict['voxel_num_points'], batch_dict[
                'voxel_coords'] = voxels, voxel_num_points, voxel_coords
        else:
            zeros = torch.zeros_like(batch_dict['voxels'][..., 0:self.code_num_dim], device="cuda")
            batch_dict["added_occ_b_ind"] = torch.zeros([1], dtype=torch.int64, device="cuda")
            batch_dict["added_occ_xyz"] = torch.zeros([1, 3], dtype=torch.float32, device="cuda")
            batch_dict["occ_pnts"] = torch.zeros([1, 4], dtype=torch.float32, device="cuda")
            batch_dict['voxels'] = torch.cat((batch_dict['voxels'], zeros), axis=-1)

        if not self.pass_gradient:
            batch_dict['occ_pnts'] = batch_dict['occ_pnts'].detach()
            batch_dict['added_occ_xyz'] = batch_dict['added_occ_xyz'].detach()
            batch_dict['added_occ_b_ind'] = batch_dict['added_occ_b_ind'].detach()
            batch_dict['voxels'] = batch_dict['voxels'].detach()

        return batch_dict
