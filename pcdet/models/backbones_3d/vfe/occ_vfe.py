import torch

from tools.visual_utils.open3d_vis_utils import draw_scenes, draw_scenes_voxel_b, draw_scenes_voxel_a
from .vfe_template import VFETemplate


class OccVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, data_cfg, **kwargs):
        super().__init__(model_cfg=model_cfg)
        self.num_point_features = num_point_features
        self.maxprob = kwargs["maxprob"]
        self.num_raw_features = len(data_cfg.POINT_FEATURE_ENCODING.used_feature_list) # DATA_AUGMENTOR.AUG_CONFIG_LIST[0].get('NUM_POINT_FEATURES', None)

    def get_output_feature_dim(self):
        return self.num_point_features

    def get_paddings_indicator(self, actual_num, max_num, axis=0):
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num_range = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        paddings_indicator = max_num_range < actual_num.int()
        return paddings_indicator

    def forward(self, batch_dict, **kwargs):
        """
        Args:
            batch_dict:
                voxels: (num_voxels, max_points_per_voxel, C)
                voxel_num_points: optional (num_voxels)
            **kwargs:

        Returns:
            vfe_features: (num_voxels, C)
        """
        # voxel_features代表每个voxel内最多4个points，每个points维度6代表xyz反射+预测概率+有预测概率就是1，因此大批量的点最后两维是0
        voxel_features, voxel_num_points, voxel_coords = batch_dict['voxels'], batch_dict['voxel_num_points'], batch_dict['voxel_coords']
        # draw_scenes(batch_dict['points'][:, 1:], gt_boxes=batch_dict['gt_boxes'][0])
        # draw_scenes_voxel_a(voxel_coords)
        # draw_scenes(voxel_features.reshape(-1, 6), gt_boxes=batch_dict['gt_boxes'][0])
        voxel_count = voxel_features.shape[1]  # 一个voxel内最多4个点云
        mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)  # 因为voxel放了4个点云，找真正不空的地方
        raw_mask = (voxel_features[:, :, -1] < 0.05) & mask  # 真正有点的位置中，没有被占有的点
        occ_mask = (voxel_features[:, :, -1] >= 0.05) & mask  # 被占有的点
        raw_normalizer = raw_mask[:, :].sum(dim=1, keepdim=False).view(-1, 1)
        occ_normalizer = occ_mask[:, :].sum(dim=1, keepdim=False).view(-1, 1)
        occ_voxel_mask = (occ_normalizer > 0.5) & (raw_normalizer < 0.5)
        raw_normalizer = torch.clamp_min(raw_normalizer, min=1.0).type_as(voxel_features)
        occ_normalizer = torch.clamp_min(occ_normalizer, min=1.0).type_as(voxel_features)

        voxel_features_raw = (raw_mask.unsqueeze(-1) * voxel_features[:, :, :self.num_raw_features]).sum(dim=1, keepdim=False) / raw_normalizer
        voxel_features_occ = (occ_mask.unsqueeze(-1) * voxel_features[:, :, :self.num_raw_features]).sum(dim=1, keepdim=False) / occ_normalizer

        # draw_scenes(voxel_features_raw)
        batch_dict['voxel_features'] = voxel_features_raw + occ_voxel_mask * voxel_features_occ  # 【17970 4】原始点云+完整形状点云
        # draw_scenes(batch_dict['voxel_features'])
        occ_max = voxel_features[:, :, self.num_raw_features:].max(dim=1, keepdim=False)[0]  # 每个体素的占有概率
        batch_dict['voxel_features'] = torch.cat([batch_dict['voxel_features'], occ_max], dim=-1)  # 【17970 6】
        # draw_scenes(batch_dict['voxel_features'])
        batch_dict['occ_voxel_features'] = occ_max  # [占有率,1]  17970 2
        # print("voxel_features", torch.min(batch_dict['voxel_features'],dim=0)[0], torch.max(batch_dict['voxel_features'],dim=0)[0])
        return batch_dict

