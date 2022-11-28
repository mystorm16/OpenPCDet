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
        # 原始点特征转2D特征
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

        if self.training == True:
            # bm points转2D BEV图
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

            # 可视化bev shape mask
            # vis = bm_batch_spatial_features.reshape(496, 432)
            # fig = plt.figure(figsize=(6, 6))
            # sns.heatmap(vis.cpu().numpy())
            # plt.show()

            # car bm points转2D BEV图
            batch_dict['c_bm_pillar_features'] = torch.ones(  # 先创建一个N*1的全1 tensor
                batch_dict['c_bm_voxels'].shape[0], 1,
                dtype=pillar_features.dtype,
                device=pillar_features.device)
            c_bm_pillar_features, c_bm_coords = batch_dict['c_bm_pillar_features'], batch_dict['c_bm_voxel_coords']
            c_bm_batch_spatial_features = []
            for batch_idx in range(batch_size):
                c_bm_spatial_feature = torch.zeros(  # 整个场景（496*432）的全0tensor 用一维索引编码
                    1, self.nz * self.nx * self.ny,
                    dtype=pillar_features.dtype,
                    device=pillar_features.device)

                batch_mask = c_bm_coords[:, 0] == batch_idx
                this_coords = c_bm_coords[batch_mask, :]
                indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]  # bm points对应的一维索引号
                indices = indices.type(torch.long)
                c_bm_pillars = c_bm_pillar_features[batch_mask, :]
                c_bm_pillars = c_bm_pillars.t()
                c_bm_spatial_feature[:, indices] = c_bm_pillars
                c_bm_batch_spatial_features.append(c_bm_spatial_feature)

            c_bm_batch_spatial_features = torch.stack(c_bm_batch_spatial_features, 0)
            c_bm_batch_spatial_features = c_bm_batch_spatial_features.view(batch_size, 1 * self.nz, self.ny, self.nx)
            batch_dict['c_bm_spatial_features'] = c_bm_batch_spatial_features

            # 可视化car bev shape mask
            # vis = c_bm_batch_spatial_features.reshape(496, 432)
            # fig = plt.figure(figsize=(6, 6))
            # sns.heatmap(vis.cpu().numpy())
            # plt.show()

            # ped bm points转2D BEV图
            batch_dict['p_bm_pillar_features'] = torch.ones(  # 先创建一个N*1的全1 tensor
                batch_dict['p_bm_voxels'].shape[0], 1,
                dtype=pillar_features.dtype,
                device=pillar_features.device)
            p_bm_pillar_features, p_bm_coords = batch_dict['p_bm_pillar_features'], batch_dict['p_bm_voxel_coords']
            p_bm_batch_spatial_features = []
            for batch_idx in range(batch_size):
                p_bm_spatial_feature = torch.zeros(  # 整个场景（496*432）的全0tensor 用一维索引编码
                    1, self.nz * self.nx * self.ny,
                    dtype=pillar_features.dtype,
                    device=pillar_features.device)

                batch_mask = p_bm_coords[:, 0] == batch_idx
                this_coords = p_bm_coords[batch_mask, :]
                indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]  # bm points对应的一维索引号
                indices = indices.type(torch.long)
                p_bm_pillars = p_bm_pillar_features[batch_mask, :]
                p_bm_pillars = p_bm_pillars.t()
                p_bm_spatial_feature[:, indices] = p_bm_pillars
                p_bm_batch_spatial_features.append(p_bm_spatial_feature)

            p_bm_batch_spatial_features = torch.stack(p_bm_batch_spatial_features, 0)
            p_bm_batch_spatial_features = p_bm_batch_spatial_features.view(batch_size, 1 * self.nz, self.ny, self.nx)
            batch_dict['p_bm_spatial_features'] = p_bm_batch_spatial_features

            # 可视化car bev shape mask
            # vis = p_bm_batch_spatial_features.reshape(496, 432)
            # fig = plt.figure(figsize=(6, 6))
            # sns.heatmap(vis.cpu().numpy())
            # plt.show()

            # cyc bm points转2D BEV图
            batch_dict['cy_bm_pillar_features'] = torch.ones(  # 先创建一个N*1的全1 tensor
                batch_dict['cy_bm_voxels'].shape[0], 1,
                dtype=pillar_features.dtype,
                device=pillar_features.device)
            cy_bm_pillar_features, cy_bm_coords = batch_dict['cy_bm_pillar_features'], batch_dict['cy_bm_voxel_coords']
            cy_bm_batch_spatial_features = []
            for batch_idx in range(batch_size):
                cy_bm_spatial_feature = torch.zeros(  # 整个场景（496*432）的全0tensor 用一维索引编码
                    1, self.nz * self.nx * self.ny,
                    dtype=pillar_features.dtype,
                    device=pillar_features.device)

                batch_mask = cy_bm_coords[:, 0] == batch_idx
                this_coords = cy_bm_coords[batch_mask, :]
                indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]  # bm points对应的一维索引号
                indices = indices.type(torch.long)
                cy_bm_pillars = cy_bm_pillar_features[batch_mask, :]
                cy_bm_pillars = cy_bm_pillars.t()
                cy_bm_spatial_feature[:, indices] = cy_bm_pillars
                cy_bm_batch_spatial_features.append(cy_bm_spatial_feature)

            cy_bm_batch_spatial_features = torch.stack(cy_bm_batch_spatial_features, 0)
            cy_bm_batch_spatial_features = cy_bm_batch_spatial_features.view(batch_size, 1 * self.nz, self.ny, self.nx)
            batch_dict['cy_bm_spatial_features'] = cy_bm_batch_spatial_features

            # 可视化car bev shape mask
            # vis = cy_bm_batch_spatial_features.reshape(496, 432)
            # fig = plt.figure(figsize=(6, 6))
            # sns.heatmap(vis.cpu().numpy())
            # plt.show()

        # 1 64 496 432 + 1 1 496 432 -> 1 65 496 432
        # batch_dict['spatial_features'] = torch.cat((batch_dict['spatial_features'], batch_dict['bm_spatial_features']), dim=1)

        return batch_dict
