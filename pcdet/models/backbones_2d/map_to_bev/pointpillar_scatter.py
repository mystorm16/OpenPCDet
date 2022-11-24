import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import seaborn as sns;
sns.set()

class PointPillarScatter(nn.Module):
    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()

        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.nx, self.ny, self.nz = grid_size
        assert self.nz == 1

    def forward(self, batch_dict, **kwargs):
        pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']
        batch_spatial_features = []
        batch_size = coords[:, 0].max().int().item() + 1
        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                # self.num_bev_features,
                64,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)

            batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            pillars = pillar_features[batch_mask, :]
            pillars = pillars.t()
            spatial_feature[:, indices] = pillars
            batch_spatial_features.append(spatial_feature)

        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        # batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features * self.nz, self.ny, self.nx)
        batch_spatial_features = batch_spatial_features.view(batch_size, 64 * self.nz, self.ny, self.nx)
        batch_dict['spatial_features'] = batch_spatial_features

        # dense的bm特征 转 sparse
        batch_dict['bm_pillar_features'] = torch.ones(  # 先创建一个N*1的全1 tensor
            batch_dict['bm_voxels'].shape[0], 1,
            dtype=pillar_features.dtype,
            device=pillar_features.device)
        bm_pillar_features, bm_coords = batch_dict['bm_pillar_features'], batch_dict['bm_voxel_coords']
        bm_batch_spatial_features = []
        for batch_idx in range(batch_size):
            bm_spatial_feature = torch.zeros(  # 整个场景（496*432）的全0tensor 用一维索引编码
                1, self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)

            batch_mask = bm_coords[:, 0] == batch_idx
            this_coords = bm_coords[batch_mask, :]
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]  # bm points对应的一维索引号
            indices = indices.type(torch.long)
            bm_pillars = bm_pillar_features[batch_mask, :]
            bm_pillars = bm_pillars.t()
            bm_spatial_feature[:, indices] = bm_pillars
            bm_batch_spatial_features.append(bm_spatial_feature)

        bm_batch_spatial_features = torch.stack(bm_batch_spatial_features, 0)
        bm_batch_spatial_features = bm_batch_spatial_features.view(batch_size, 1 * self.nz, self.ny, self.nx)
        batch_dict['bm_spatial_features'] = bm_batch_spatial_features

        # 1 64 496 432 + 1 1 496 432 -> 1 65 496 432
        batch_dict['spatial_features'] = torch.cat((batch_dict['spatial_features'], batch_dict['bm_spatial_features']), dim=1)

        # 可视化bev shape mask
        # vis = bm_batch_spatial_features.reshape(496, 432)
        # fig = plt.figure(figsize=(6, 6))
        # sns.heatmap(vis.cpu().numpy())
        # plt.show()
        return batch_dict
