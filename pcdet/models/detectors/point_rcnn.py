from .detector3d_template import Detector3DTemplate

OPEN3D_FLAG = True


class PointRCNN(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
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
            pred_dicts, recall_dicts = self.post_processing(batch_dict)  # 一个batch的predict box、score、labelpred_dicts里

            # sy add
            # iou_list, points_num_list = iou_ptsnum(batch_dict, pred_dicts)
            # recall_dicts.update(iou=iou_list, points_num=points_num_list)

            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}
        loss_point, tb_dict = self.point_head.get_loss()
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)

        loss = loss_point + loss_rcnn
        return loss, tb_dict, disp_dict
