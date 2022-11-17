import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns;

sns.set()
from tools.visual_utils.open3d_vis_utils import draw_scenes, \
    draw_scenes_voxel_b, draw_scenes_voxel_a, draw_spherical_voxels_points, draw_scenes_points2voxel, \
    draw_spherical_voxels_index, draw_spherical_voxels_4
# from tools.visual_utils.visualize_utils import draw_spherical_voxels

from .occ_targets_template import OccTargetsTemplate
from ....utils import coords_utils, point_box_utils, common_utils


class OccTargets3D(OccTargetsTemplate):
    def __init__(self, model_cfg, voxel_size, point_cloud_range, data_cfg, grid_size,
                 num_class, voxel_centers):
        super().__init__(model_cfg, voxel_size, point_cloud_range, data_cfg, grid_size,
                         num_class, voxel_centers)
        self.reg = model_cfg.PARAMS.get("REG", False)

    def create_predict_area(self, voxel_bnysynxsxnzsz, voxel_num_points_float, batch_size, batch_dict):
        return self.create_predict_area2d(voxel_bnysynxsxnzsz, voxel_num_points_float, batch_size, batch_dict)

    def forward(self, batch_dict, **kwargs):
        # 可视化
        # draw_scenes(batch_dict['points'][:, 1:], gt_boxes=batch_dict['gt_boxes'][0])  # 原始点云
        # draw_scenes(batch_dict['bm_points'][:, 1:], gt_boxes=batch_dict['gt_boxes'][0])  # 补全后的shape
        # draw_scenes(batch_dict['occ_voxels'].reshape(-1, 4))  # occ体素系下的原始点云，一个体素最多12点
        # draw_scenes_voxel_a(batch_dict['occ_voxel_coords'])  # occ体素系下不为空的体素
        # draw_scenes(batch_dict['det_voxels'].reshape(-1, 4))  # det体素系下的原始点云，一个体素最多5个点
        # draw_scenes_voxel_a(batch_dict['det_voxel_coords'])  # det体素下不为空的体素

        # 取出occ系下的数据 voxel_features是每个voxel内的原始点坐标
        voxel_features, voxel_num_points, coords = batch_dict['occ_voxels'], \
                                                   batch_dict['occ_voxel_num_points'], batch_dict['occ_voxel_coords']
        # draw_scenes(voxel_features.reshape(-1, 4))
        # draw_scenes_voxel_a(coords)

        voxel_count = voxel_features.shape[1]  # 每个voxel内最多有12个point
        mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)  # 每个voxel内实际是否有点的mask
        batch_dict["voxel_point_mask"] = mask  # 【7502 12】真正有点的那些位置
        # 创建真值label
        batch_dict = self.create_voxel_res_label(batch_dict, mask) \
            if self.reg else self.create_voxel_label(batch_dict, mask)
        batch_dict["final_point_mask"] = mask  # 真正存在点的那些位置
        return batch_dict

    # 给体素打结果标签
    def create_voxel_res_label(self, batch_dict, valid_mask):
        # occ_pnts把sphere系下voxel内的原始点坐标转为直角系坐标,相较于原始点云部分位置没有那么密了
        occ_pnts = batch_dict['occ_voxels']  # [N 12 4]
        batch_dict['voxels'] = occ_pnts  # 直角坐标存入 voxels
        #  voxel_features把sphere系下voxel内的原始点坐标转为直角系坐标、voxel_coords还是sphere系voxel的index
        voxel_features, voxel_coords, gt_boxes_num, gt_boxes, bs = occ_pnts, batch_dict['occ_voxel_coords'], batch_dict[
            "gt_boxes_num"], batch_dict["gt_boxes"], batch_dict["gt_boxes"].shape[0]
        if self.num_class == 1:
            gt_label = (gt_boxes[..., -1:] > 1e-2).to(torch.float32)
            gt_boxes = torch.cat([gt_boxes[..., :-1], gt_label], dim=-1)

        # 将occ体素内的有效点展开，因为一个sphere体素内最多12个原始点云，存储结构中12里面有很多是无效的
        # valid_coords_bnznynx代表展开后的原始点云对应的occ体素index、valid_voxel_features为展开后的原始点云
        valid_coords_bnznynx, valid_voxel_features = self.get_valid(valid_mask, voxel_coords,
                                                                    voxel_features)  # 按体素内的每个点展开
        # draw_scenes_voxel_a(valid_coords_bnznynx)
        # draw_scenes(valid_voxel_features, gt_boxes=batch_dict['gt_boxes'][0])
        voxelwise_mask = self.get_voxelwise_mask(valid_coords_bnznynx, bs)  # 把原始点云coord转到1*9*157*209的sphere coord

        # vcc area
        # vcc_mask = self.create_predict_area3d(bs, valid_coords_bnznynx)  # 对每个原始coord进行9*5*5的展开  还是sphere coord
        # occ_voxelwise_mask = vcc_mask

        # center area
        # center_area = batch_dict['center_area']
        # occ_voxelwise_mask = self.create_center_area3d(center_area, batch_dict)  # occ_voxelwise_mask [1 9 176 200] 是信息缺失区域

        # center vcc
        center = batch_dict['final_centers']
        occ_voxelwise_mask = self.create_center_vcc(center, batch_dict)  # occ_voxelwise_mask [1 9 176 200] 是信息缺失区域

        # draw_scenes_voxel_b(occ_voxelwise_mask)

        # gt box内的前景点mask
        fore_voxelwise_mask, fore_res_mtrx, mirr_fore_voxelwise_mask, mirr_res_mtrx = self.get_fore_mirr_voxelwise_mask_res(
            batch_dict, bs, valid_coords_bnznynx, valid_voxel_features, gt_boxes_num, gt_boxes)
        # draw_scenes_voxel_b(fore_voxelwise_mask)

        # gt box内的前景点对称后的mask
        # draw_scenes_voxel_b(mirr_fore_voxelwise_mask)
        mirr_fore_voxelwise_mask = mirr_fore_voxelwise_mask * (1 - voxelwise_mask)  # exclude original occupied
        # draw_scenes_voxel_b(mirr_fore_voxelwise_mask)

        mirr_res_mtrx = mirr_res_mtrx * (1 - voxelwise_mask).unsqueeze(1)

        if self.model_cfg.TARGETS.TMPLT:
            bm_voxelwise_mask, bm_res_mtrx = self.get_bm_voxelwise_mask_res(batch_dict, bs, gt_boxes_num, gt_boxes)
            bm_voxelwise_mask = bm_voxelwise_mask * (1 - voxelwise_mask) * (1 - mirr_fore_voxelwise_mask)  # 完整形状点云除去原始点和前景对称点
            # draw_scenes_voxel_b(bm_voxelwise_mask)
            # bm_res_mtrx是补全点的三维偏移量 在回归的时候使用
            bm_res_mtrx = bm_res_mtrx * (1 - voxelwise_mask).unsqueeze(1) * (1 - mirr_fore_voxelwise_mask).unsqueeze(1)
        else:
            bm_voxelwise_mask = torch.zeros_like(voxelwise_mask, dtype=voxelwise_mask.dtype,
                                                 device=voxelwise_mask.device)

        '''forebox_label:gt内的2D点拉长'''
        forebox_label = None
        if self.data_cfg.OCC.BOX_WEIGHT != 1.0:
            bs, max_num_box, box_c = list(gt_boxes.shape)
            forebox_label = torch.zeros([bs, self.nz, self.ny, self.nx], dtype=torch.int8, device="cuda")
            shift = torch.tensor(np.asarray([[0.0, 0.0, 0.0]]), device="cuda", dtype=torch.float32)

            for i in range(bs):
                cur_gt_boxes = gt_boxes[i, :gt_boxes_num[i]]

                all_voxel_centers_2d = point_box_utils.rotatez(self.all_voxel_centers_2d, batch_dict["rot_z"][i]) \
                    if "rot_z" in batch_dict else self.all_voxel_centers_2d

                voxel_box_label2d = point_box_utils.torch_points_in_box_2d_mask(all_voxel_centers_2d, cur_gt_boxes,
                                                                                shift=shift[..., :2]).view(self.ny,
                                                                                                           self.nx).nonzero()
                # 可视化all_voxel_centers_2d
                # fig = plt.figure(figsize=(6, 6))
                # plt.plot(all_voxel_centers_2d[:, 0].cpu(), all_voxel_centers_2d[:, 1].cpu())
                # plt.show()

                if voxel_box_label2d.shape[0] > 0:  # 把gt box内的2D点 拉长成三维 all_voxel_centers_filtered
                    all_voxel_centers_filtered = self.all_voxel_centers[:, voxel_box_label2d[:, 0],
                                                 voxel_box_label2d[:, 1], ...].reshape(-1, 3)
                    # draw_scenes(all_voxel_centers_filtered)
                    if "rot_z" in batch_dict:
                        # draw_scenes(all_voxel_centers_filtered)
                        all_voxel_centers_filtered = point_box_utils.rotatez(all_voxel_centers_filtered, batch_dict["rot_z"][i])
                        # draw_scenes(all_voxel_centers_filtered)
                    voxel_box_label = point_box_utils.torch_points_in_box_3d_label(
                        all_voxel_centers_filtered, cur_gt_boxes, gt_boxes_num[i], shift=shift)[
                        0]  # 补全的点是否在gt box内的mask
                    forebox_label[i, :, voxel_box_label2d[:, 0], voxel_box_label2d[:, 1]] = voxel_box_label.view(
                        self.nz, -1)  # gt内的2D 拉长后的形状
                    # draw_scenes_voxel_b(forebox_label)
        # 生成补全后的shape，作为真值监督训练，这里是cls 存在shape的位置是1否则是0
        batch_dict = self.prepare_cls_loss_map(batch_dict, voxelwise_mask, occ_voxelwise_mask,
                                               fore_voxelwise_mask, mirr_fore_voxelwise_mask, bm_voxelwise_mask,
                                               forebox_label=forebox_label)  # 得到cls weight label

        batch_dict = self.prepare_reg_loss_map(batch_dict, fore_res_mtrx, mirr_res_mtrx, bm_res_mtrx)  # 得到reg weight label

        # draw_scenes_voxel_b(batch_dict['general_cls_loss_mask'])
        # draw_scenes_voxel_b(batch_dict['general_reg_loss_mask'])
        return batch_dict

    def get_bm_voxelwise_mask_res(self, batch_dict, bs, gt_boxes_num, gt_boxes):
        bm_voxelwise_mask = torch.zeros([bs, self.nz, self.ny, self.nx], dtype=torch.uint8, device="cuda")
        if "bm_points" in batch_dict and len(batch_dict["bm_points"]) > 0:
            bm_binds, bm_carte_points = batch_dict["bm_points"][..., 0:1].to(torch.int64), batch_dict["bm_points"][..., 1:]
            label_array = torch.nonzero(
                point_box_utils.torch_points_in_box_3d_label_batch(bm_carte_points, bm_binds, gt_boxes, gt_boxes_num,
                                                                   bs))[..., 0]
            bm_binds = bm_binds[..., 0][label_array]
            bm_carte_points = bm_carte_points[label_array, :]
            occ_coords_bm_points = bm_carte_points  # 处理掉一些在gtbox之外的补全点
            if "rot_z" in batch_dict:
                rot_z = batch_dict["rot_z"][bm_binds]
                noise_rotation = -rot_z * np.pi / 180
                occ_coords_bm_points = common_utils.rotate_points_along_z(occ_coords_bm_points.unsqueeze(1),
                                                                          noise_rotation).squeeze(1)
            inrange_coords_bm, inrange_inds_bm = self.point2coords_inrange(occ_coords_bm_points,
                                                                           self.point_origin_tensor,
                                                                           self.point_max_tensor, self.max_grid_tensor,
                                                                           self.min_grid_tensor, self.voxel_size)
            bm_coords = torch.cat([bm_binds[inrange_inds_bm].unsqueeze(-1), self.xyz2zyx(inrange_coords_bm)], dim=-1)
            bm_res_mtrx = self.get_mean_res(bm_carte_points[inrange_inds_bm], bm_coords, bs, self.nz, self.ny, self.nx,
                                            batch_dict, rot=True)
            bm_voxelwise_mask[
                bm_coords[..., 0], bm_coords[..., 1], bm_coords[..., 2], bm_coords[..., 3]] = torch.ones_like(
                bm_coords[..., 0], dtype=torch.uint8, device=bm_voxelwise_mask.device)  ##
        else:
            bm_res_mtrx = torch.zeros([bs, 3, self.nz, self.ny, self.nx], dtype=torch.float32, device="cuda")

        return bm_voxelwise_mask, bm_res_mtrx

    def get_mean_res(self, feat, coords, bs, nz, ny, nx, batch_dict, rot=False):
        xyz_spatial = torch.zeros([bs, 3, nz, ny, nx], dtype=torch.float32, device="cuda")
        if len(coords) > 0:
            uni_coords, inverse_indices, labels_count = torch.unique(coords, return_inverse=True, return_counts=True,
                                                                     dim=0)
            mean_xyz = torch.zeros([uni_coords.shape[0], 3], dtype=feat.dtype, device=feat.device).scatter_add_ \
                           (0, inverse_indices.view(inverse_indices.size(0), 1).expand(-1, 3),
                            feat[..., :3]) / labels_count.float().unsqueeze(1)
            # mean_xyz = torch_scatter.scatter_mean(feat[..., :3], inverse_indices, dim=0)
            # draw_scenes(mean_xyz)
            mean_xyz -= self.get_voxel_center_xyz(uni_coords, batch_dict, rot=True)  # 得到每个点相对于voxel中心的的偏移量
            # draw_scenes(mean_xyz)
            xyz_spatial[uni_coords[..., 0], :, uni_coords[..., 1], uni_coords[..., 2], uni_coords[..., 3]] = mean_xyz
        return xyz_spatial

    # 从coords得每个voxel的中心点坐标
    def get_voxel_center_xyz(self, coords, batch_dict, rot=True):
        # draw_scenes_voxel_a(coords)
        voxel_centers = (coords[:, [3, 2, 1]].float() + 0.5) * self.voxel_size + self.point_origin_tensor  # +0.5把散点移到原心处
        # draw_scenes(voxel_centers)
        if "rot_z" in batch_dict and rot:
            rot_z = batch_dict["rot_z"][coords[:, 0]]
            noise_rotation = rot_z * np.pi / 180
            voxel_centers = common_utils.rotate_points_along_z(voxel_centers.unsqueeze(1), noise_rotation).squeeze(
                1)
            # draw_scenes(voxel_centers)
        return voxel_centers  # 每个voxel的中心点坐标

    def get_fore_mirr_voxelwise_mask_res(self, batch_dict, bs, valid_coords_bnznynx, valid_voxel_features, gt_boxes_num,
                                         gt_boxes):
        fore_voxelwise_mask, mirr_fore_voxelwise_mask = [
            torch.zeros([bs, self.nz, self.ny, self.nx], dtype=torch.uint8, device="cuda") for i in range(2)]
        fore_inds, mirr_inbox_point, mirr_binds = point_box_utils.torch_points_and_sym_in_box_3d_batch(
            valid_voxel_features[..., :3], valid_coords_bnznynx, gt_boxes, gt_boxes_num, bs,
            batch_dict['box_mirr_flag'])
        # draw_scenes(mirr_inbox_point)
        fore_coords = valid_coords_bnznynx[fore_inds]  # b zyx
        # draw_scenes_voxel_a(fore_coords)
        fore_voxelwise_mask[
            fore_coords[..., 0], fore_coords[..., 1], fore_coords[..., 2], fore_coords[..., 3]] = torch.ones_like(
            fore_coords[..., 0], dtype=torch.uint8, device=fore_voxelwise_mask.device)
        fore_res_mtrx = self.get_mean_res(valid_voxel_features[fore_inds], fore_coords, bs, self.nz, self.ny, self.nx,
                                          batch_dict, rot=True)  # 每个原始点相对于voxel中心的偏移量
        mirr_res_mtrx = torch.zeros([bs, 3, self.nz, self.ny, self.nx], device=fore_voxelwise_mask.device,
                                    dtype=torch.float32)
        if mirr_inbox_point is not None:
            occ_coords_mirr_points = mirr_inbox_point
            if "rot_z" in batch_dict:
                rot_z = batch_dict["rot_z"][mirr_binds]
                noise_rotation = -rot_z * np.pi / 180
                occ_coords_mirr_points = common_utils.rotate_points_along_z(occ_coords_mirr_points.unsqueeze(1),
                                                                            noise_rotation).squeeze(1)

            inrange_coords_mirr, inrange_inds_mirr = self.point2coords_inrange(occ_coords_mirr_points,
                                                                               self.point_origin_tensor,
                                                                               self.point_max_tensor,
                                                                               self.max_grid_tensor,
                                                                               self.min_grid_tensor, self.voxel_size)
            mirr_coords = torch.cat([mirr_binds[inrange_inds_mirr].unsqueeze(-1), self.xyz2zyx(inrange_coords_mirr)],
                                    dim=-1)  # mirror sphere b z y x
            mirr_res_mtrx = self.get_mean_res(mirr_inbox_point[inrange_inds_mirr], mirr_coords, bs, self.nz, self.ny,
                                              self.nx, batch_dict, rot=True)

            mirr_fore_voxelwise_mask[
                mirr_coords[..., 0], mirr_coords[..., 1], mirr_coords[..., 2], mirr_coords[..., 3]] = torch.ones_like(
                mirr_coords[..., 0], dtype=torch.uint8, device=mirr_fore_voxelwise_mask.device)

        return fore_voxelwise_mask, fore_res_mtrx, mirr_fore_voxelwise_mask, mirr_res_mtrx
