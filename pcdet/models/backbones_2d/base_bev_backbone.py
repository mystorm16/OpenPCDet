import time

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import seaborn as sns;
sns.set()

class BaseBEVBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg

        if self.model_cfg.get('LAYER_NUMS', None) is not None:
            assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.LAYER_STRIDES) == len(self.model_cfg.NUM_FILTERS)
            layer_nums = self.model_cfg.LAYER_NUMS
            layer_strides = self.model_cfg.LAYER_STRIDES
            num_filters = self.model_cfg.NUM_FILTERS
        else:
            layer_nums = layer_strides = num_filters = []

        if self.model_cfg.get('UPSAMPLE_STRIDES', None) is not None:
            assert len(self.model_cfg.UPSAMPLE_STRIDES) == len(self.model_cfg.NUM_UPSAMPLE_FILTERS)
            num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
            upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        else:
            upsample_strides = num_upsample_filters = []

        num_levels = len(layer_nums)
        c_in_list = [input_channels, *num_filters[:-1]]
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1),  # 图像四周都填充0
                nn.Conv2d(
                    in_channels=c_in_list[idx]+self.model_cfg.FUSE_BEV if self.model_cfg.get('FUSE_BEV', None) is not None else c_in_list[idx],
                    out_channels=num_filters[idx], kernel_size=3,  # +1(融了一维特征) +2(融了两维特征)
                    stride=layer_strides[idx], padding=0, bias=False
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ])
            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.num_bev_features = c_in  # sy add bev模块处理完毕的输出feature

        # bev多层特征融合
        self.bev_squeeze_blocks = nn.ModuleList()
        for idx in [1, 2, 4]:
            # bev_shape_squeeze = nn.Sequential(nn.MaxPool2d(2))
            bev_shape_squeeze = nn.Sequential(nn.ZeroPad2d(1),
                                              nn.Conv2d(1, 1, kernel_size=(3, 3), stride=idx, bias=False),
                                              nn.BatchNorm2d(1, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
                                              nn.ReLU())
            self.bev_squeeze_blocks.append(bev_shape_squeeze)

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features = data_dict['spatial_features']  # 原版特征 1 64 496 432
        ups = []
        ret_dict = {}
        x = spatial_features

        # 融合hm特征
        if data_dict.__contains__('hm_binary_fuse'):
            flag_fuse_features = True
            # data_dict['hm_binary_fuse'] = data_dict['hm_binary_fuse'].detach()
            # data_dict['hm_prob_fuse'] = data_dict['hm_prob_fuse'].detach()

            # vis = data_dict['hm_binary_fuse'][0][0].detach().cpu().numpy()
            # fig = plt.figure(figsize=(10, 10))
            # sns.heatmap(vis)
            # plt.show()
            # vis = data_dict['hm_prob_fuse'][0][0].detach().cpu().numpy()
            # fig = plt.figure(figsize=(10, 10))
            # sns.heatmap(vis)
            # plt.show()
        else:
            flag_fuse_features = False

        for i in range(len(self.blocks)):
            # bev特征融合
            if flag_fuse_features == True:
                # 融合二值化特征
                bev_squeeze_features_binary = data_dict['hm_binary_fuse']
                x_bev_binary = self.bev_squeeze_blocks[i](bev_squeeze_features_binary)  # bev特征压缩
                x = torch.cat((x, x_bev_binary), dim=1)

                # 融合预测概率特征
                bev_squeeze_features_prob = data_dict['hm_prob_fuse']
                x_bev_prob = self.bev_squeeze_blocks[i](bev_squeeze_features_prob)  # bev特征压缩
                x = torch.cat((x, x_bev_prob), dim=1)

            x = self.blocks[i](x)
            stride = int(spatial_features.shape[2] / x.shape[2])
            ret_dict['spatial_features_%dx' % stride] = x
            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x))
            else:
                ups.append(x)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)

        data_dict['spatial_features_2d'] = x

        # 特征融合
        # bev_squeeze_features = self.bev_shape_squeeze(data_dict['bm_spatial_features'])
        # data_dict['spatial_features_2d'] = torch.cat((data_dict['spatial_features_2d'], bev_squeeze_features), dim=1)
        # 可视化bev shape mask
        # vis = bev_squeeze_features.reshape(248, 216)
        # fig = plt.figure(figsize=(6, 6))
        # sns.heatmap(vis.cpu().numpy())
        # plt.show()
        return data_dict
