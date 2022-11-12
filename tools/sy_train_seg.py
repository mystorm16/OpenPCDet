import time

import _init_path
import argparse
import datetime
import glob
import os
from pathlib import Path

from pcdet.models.backbones_3d import PointNet2MSG
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
from pcdet.utils.loss_utils import SigmoidFocalClassificationLoss
from test import repeat_eval_ckpt

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, model_fn_decorator, load_data_to_gpu
from pcdet.utils import common_utils, box_utils
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from train_utils.optimization import build_optimizer, build_scheduler
from train_utils.train_utils import train_model


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')

    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=None, required=False, help='number of epochs to train for')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained_model')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--sync_bn', action='store_true', default=False, help='whether to use sync bn')
    parser.add_argument('--fix_random_seed', action='store_true', default=False, help='')
    parser.add_argument('--ckpt_save_interval', type=int, default=1, help='number of training epochs')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--max_ckpt_save_num', type=int, default=30, help='max number of saved checkpoint')
    parser.add_argument('--merge_all_iters_to_one_epoch', action='store_true', default=False, help='')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    parser.add_argument('--max_waiting_mins', type=int, default=0, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--num_epochs_to_eval', type=int, default=0, help='number of checkpoints to be evaluated')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


# 含有两个隐藏层的mlp
class Seg_MLP(nn.Module):
    def __init__(self):
        super(Seg_MLP, self).__init__()

        self.model1 = Sequential(
            nn.Linear(128, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 3, bias=True)
        )

    def forward(self, batch):
        point_cls_preds = self.model1(batch['point_features'])
        return point_cls_preds


# 预测的前景点和前景点真值的关系
def assign_targets(input_dict):
    # 得到每个点的坐标 shape（bacth * 16384, 4），其中4个维度分别是batch_id，x，y，z
    point_coords = input_dict['point_coords']
    # 取出gt_box，shape （batch， num_of_GTs, 8），其中维度8表示 x, y, z, l, w, h, heading, class_id
    gt_boxes = input_dict['gt_boxes']
    assert gt_boxes.shape.__len__() == 3, 'gt_boxes.shape=%s' % str(gt_boxes.shape)
    assert point_coords.shape.__len__() in [2], 'points.shape=%s' % str(point_coords.shape)
    # 把box label扩大一些
    batch_size = gt_boxes.shape[0]
    extend_gt_boxes = box_utils.enlarge_box3d(
        gt_boxes.view(-1, gt_boxes.shape[-1]), extra_width=[0.2, 0.2, 0.2]
    ).view(batch_size, -1, gt_boxes.shape[-1])

    targets_dict = assign_stack_targets(
        points=point_coords, gt_boxes=gt_boxes, extend_gt_boxes=extend_gt_boxes,
        set_ignore_flag=True, use_ball_constraint=False,
        ret_part_labels=False, ret_box_labels=True
    )
    return targets_dict


"""                                                                         
assign_stack_targets函数完成了一批数据中所有点的前背景分配，                                    
并为每个前景点分配了对应的类别和box的7个回归参数，xyzlwhθ                                          
"""


def assign_stack_targets(points, gt_boxes, extend_gt_boxes=None,
                         ret_box_labels=False, ret_part_labels=False,
                         set_ignore_flag=True, use_ball_constraint=False, central_radius=2.0):
    assert len(points.shape) == 2 and points.shape[1] == 4, 'points.shape=%s' % str(points.shape)
    assert len(gt_boxes.shape) == 3 and gt_boxes.shape[2] == 8, 'gt_boxes.shape=%s' % str(gt_boxes.shape)
    assert extend_gt_boxes is None or len(extend_gt_boxes.shape) == 3 and extend_gt_boxes.shape[2] == 8, \
        'extend_gt_boxes.shape=%s' % str(extend_gt_boxes.shape)
    assert set_ignore_flag != use_ball_constraint, 'Choose one only!'
    # 得到一批数据中batch_size的大小，以方便逐帧完成target assign
    batch_size = gt_boxes.shape[0]
    # 得到一批数据中所有点的batch_id
    bs_idx = points[:, 0]
    # 初始化每个点云的类别，默认全0，背景点； shape （batch * 16384）
    point_cls_labels = points.new_zeros(points.shape[0]).long()

    # 逐帧点云数据进行处理
    for k in range(batch_size):
        # 得到一个mask，用于取出一批数据中属于当前帧的点     mask可理解为元素为True/False的tensor
        bs_mask = (bs_idx == k)
        # 取出一帧内对应的点
        points_single = points[bs_mask][:, 1:4]
        # 初始化当前帧中点的类别
        point_cls_labels_single = point_cls_labels.new_zeros(bs_mask.sum())
        # 计算哪些点在GTbox中, box_idxs_of_pts
        box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
            points_single.unsqueeze(dim=0), gt_boxes[k:k + 1, :, 0:7].contiguous()
        ).long().squeeze(dim=0)
        # box_fg_flag取0 1 2 3的mask，-1代表ignore
        box_fg_flag = (box_idxs_of_pts >= 0)
        # 是否忽略在enlarge box中的点 True
        if set_ignore_flag:
            # 计算哪些点在GTbox_enlarge中
            extend_box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                points_single.unsqueeze(dim=0), extend_gt_boxes[k:k + 1, :, 0:7].contiguous()
            ).long().squeeze(dim=0)
            # 前景点
            fg_flag = box_fg_flag
            # ^为异或运算符，不同为真，相同为假，这样就可以得到真实GT enlarge后的的点了
            ignore_flag = fg_flag ^ (extend_box_idxs_of_pts >= 0)
            # 将这些真实GT边上的点设置为-1      loss计算时，不考虑这类点
            point_cls_labels_single[ignore_flag] = -1
        elif use_ball_constraint:
            box_centers = gt_boxes[k][box_idxs_of_pts][:, 0:3].clone()
            box_centers[:, 2] += gt_boxes[k][box_idxs_of_pts][:, 5] / 2
            ball_flag = ((box_centers - points_single).norm(dim=1) < central_radius)
            fg_flag = box_fg_flag & ball_flag
        else:
            raise NotImplementedError

        # [box_idxs_of_pts[fg_flag]]取出所有点中属于前景的点，并为这些点分配对应的GT_box shape (num_of_gt_match_by_points, 8)
        # 8个维度分别是x, y, z, l, w, h, heading, class_id
        gt_box_of_fg_points = gt_boxes[k][box_idxs_of_pts[fg_flag]]
        num_class = 3
        # 将类别信息赋值给对应的前景点 (16384, )
        point_cls_labels_single[fg_flag] = 1 if num_class == 1 else gt_box_of_fg_points[:, -1].long()
        # 赋值点的类别GT结果到的batch中对应的帧位置
        point_cls_labels[bs_mask] = point_cls_labels_single
        return point_cls_labels


def get_cls_layer_loss(point_cls_labels, point_cls_preds):
    # 第一阶段点的GT类别
    point_cls_labels = point_cls_labels.view(-1)  # 0为背景，1,2,3分别为前景，-1不关注
    # 第一阶段点的预测类别
    point_cls_preds = point_cls_preds.view(-1, 3)
    # 取出属于前景的点的mask，0为背景，1,2,3分别为前景，-1不关注
    positives = (point_cls_labels > 0)
    # 背景点分类权重置0
    negative_cls_weights = (point_cls_labels == 0) * 1.0
    # 前景点分类权重置0
    cls_weights = (negative_cls_weights + 1.0 * positives).float()
    # 使用前景点的个数来normalize，使得一批数据中每个前景点贡献的loss一样
    pos_normalizer = positives.sum(dim=0).float()
    # 正则化每个类别分类损失权重
    cls_weights /= torch.clamp(pos_normalizer, min=1.0)

    # 初始化分类的one-hot （batch * 16384, 4）
    one_hot_targets = point_cls_preds.new_zeros(*list(point_cls_labels.shape), 3 + 1)
    # 将目标标签转换为one-hot编码形式
    one_hot_targets.scatter_(-1, (point_cls_labels * (point_cls_labels >= 0).long()).unsqueeze(dim=-1).long(), 1.0)
    # 原来背景为[1, 0, 0, 0] 现在背景为[0, 0, 0]
    one_hot_targets = one_hot_targets[..., 1:]

    # 实例化focal loss类
    cls_loss_func = SigmoidFocalClassificationLoss(alpha=0.25, gamma=2.0)
    # 计算分类损失使用focal loss
    cls_loss_src = cls_loss_func(point_cls_preds, one_hot_targets, weights=cls_weights)
    # 各类别loss置求总数
    point_loss_cls = cls_loss_src.sum()
    # 分类损失乘以分类损失权重
    point_loss_cls = point_loss_cls * 1.0

    return point_loss_cls


def main():
    args, cfg = parse_config()
    if args.launcher == 'none':
        dist_train = False
        total_gpus = 1
    else:
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_train = True

    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
        args.batch_size = args.batch_size // total_gpus

    args.epochs = cfg.OPTIMIZATION.NUM_EPOCHS if args.epochs is None else args.epochs

    if args.fix_random_seed:
        common_utils.set_random_seed(666 + cfg.LOCAL_RANK)

    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    ckpt_dir = output_dir / 'ckpt'
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / ('log_train_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # log to file
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    if dist_train:
        logger.info('total_batch_size: %d' % (total_gpus * args.batch_size))
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)
    if cfg.LOCAL_RANK == 0:
        os.system('cp %s %s' % (args.cfg_file, output_dir))

    tb_log = SummaryWriter(log_dir=str(output_dir / 'tensorboard')) if cfg.LOCAL_RANK == 0 else None

    # -----------------------create dataloader & network & optimizer---------------------------
    train_set, train_loader, train_sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_train, workers=args.workers,
        logger=logger,
        training=True,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
        total_epochs=args.epochs,
        seed=666 if args.fix_random_seed else None
    )

    # 记录训练次数
    total_train_step = 0

    model_cfg = cfg.MODEL.BACKBONE_3D
    Model_pointnet2 = PointNet2MSG(model_cfg, 4).cuda()
    seg_MLP = Seg_MLP().cuda()
    Final_Model = nn.Module()
    Final_Model.add_module('FIRST1', Model_pointnet2)
    Final_Model.add_module('SECOND2', seg_MLP)
    Final_Model = Final_Model.cuda()

    optimizer = build_optimizer(Final_Model, cfg.OPTIMIZATION)
    last_epoch = -1
    lr_scheduler, lr_warmup_scheduler = build_scheduler(
        optimizer, total_iters_each_epoch=len(train_loader), total_epochs=args.epochs,
        last_epoch=last_epoch, optim_cfg=cfg.OPTIMIZATION
    )

    # train epoch
    for i in range(10):
        dataloader_iter = iter(train_loader)
        print("-------------------第{}轮训练开始-------------------".format(i))
        for cur_data in range(len(train_loader)):
            time1 = time.perf_counter()
            batch = next(dataloader_iter)  # 从dataloader里拿time1 = time.perf_counter()数据
            time2 = time.perf_counter()
            # print("耗时1：{}".format(time2 - time1))

            load_data_to_gpu(batch)  # ndarray2tensor
            time3 = time.perf_counter()
            # print("耗时2：{}".format(time3 - time2))

            batch = Model_pointnet2(batch)  # pointnet
            time4 = time.perf_counter()
            # print("耗时3：{}".format(time4 - time3))

            point_cls_preds = seg_MLP(batch)
            # 从每个点的分类预测结果中，取出类别预测概率最大的结果  (batch * 16384, num_class) --> (batch * 16384, )
            point_cls_preds_max, _ = point_cls_preds.max(dim=-1)
            # 将类别预测分数经过sigmoid激活后放入字典中
            point_cls_scores = torch.sigmoid(point_cls_preds_max)
            time5 = time.perf_counter()
            # print("耗时4：{}".format(time5 - time4))

            # 如果在训练模式下，需要根据GTBox来生成对应的前背景点，用于点云的前背景分割，给后面计算前背景分类loss
            point_cls_labels = assign_targets(batch)

            # aa = torch.bincount(point_cls_labels)
            loss = get_cls_layer_loss(point_cls_labels, point_cls_preds)
            time6 = time.perf_counter()
            # print("耗时5：{}".format(time6 - time5))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            time7 = time.perf_counter()
            # print("耗时6：{}".format(time7 - time6))

            total_train_step = total_train_step + 1
            if total_train_step % 20 == 0:  # 训练次数每逢100 打印loss信息
                bincout1 = torch.bincount(point_cls_preds_max.reshape(-1))
                bincout2 = torch.bincount(point_cls_labels.reshape(-1))
                print("训练次数：{}，Loss：{}".format(total_train_step, loss))


if __name__ == '__main__':
    main()
