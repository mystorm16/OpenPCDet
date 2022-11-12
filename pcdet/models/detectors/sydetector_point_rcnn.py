from .detector3d_template import Detector3DTemplate


class PointRCNN_cls(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        # 在框架中增加额外模块
        self.module_topology = [
            'vfe', 'backbone_3d', 'cls_head', 'map_to_bev_module', 'pfe',
            'backbone_2d', 'dense_head', 'point_head', 'roi_head'
        ]
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        for cur_module in self.module_list:  # 循环model list进行网络预测：pointnet-pointheadbox-pointrcnnhead
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.cls_post_processing(batch_dict)
            return pred_dicts

    def cls_post_processing(self, batch_dict):
        batch_size = batch_dict['batch_size']

    def get_training_loss(self):    # get_training_loss函数 方便不同loss相加
        disp_dict = {}
        loss_point, tb_dict = self.cls_head.get_loss()

        loss = loss_point
        return loss, tb_dict, disp_dict
