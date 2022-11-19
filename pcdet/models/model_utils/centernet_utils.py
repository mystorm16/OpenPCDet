# This file is modified from https://github.com/tianweiy/CenterPoint

import torch
import torch.nn.functional as F
import numpy as np
import numba


def gaussian_radius(height, width, min_overlap=0.5):
    """
    Args:
        height: (N)
        width: (N)
        min_overlap:
    Returns:
    """
    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = (b1 ** 2 - 4 * a1 * c1).sqrt()
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = (b2 ** 2 - 4 * a2 * c2).sqrt()
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = (b3 ** 2 - 4 * a3 * c3).sqrt()
    r3 = (b3 + sq3) / 2
    ret = torch.min(torch.min(r1, r2), r3)
    return ret


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_gaussian_to_heatmap(heatmap, center, radius, k=1, valid_mask=None):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = torch.from_numpy(
        gaussian[radius - top:radius + bottom, radius - left:radius + right]
    ).to(heatmap.device).float()

    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        if valid_mask is not None:
            cur_valid_mask = valid_mask[y - top:y + bottom, x - left:x + right]
            masked_gaussian = masked_gaussian * cur_valid_mask.float()

        torch.max(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = F.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


@numba.jit(nopython=True)
def circle_nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    scores = dets[:, 2]
    order = scores.argsort()[::-1].astype(np.int32)  # highest->lowest
    ndets = dets.shape[0]
    suppressed = np.zeros((ndets), dtype=np.int32)
    keep = []
    for _i in range(ndets):
        i = order[_i]  # start with highest score box
        if suppressed[i] == 1:  # if any box have enough iou with this, remove it
            continue
        keep.append(i)
        for _j in range(_i + 1, ndets):
            j = order[_j]
            if suppressed[j] == 1:
                continue
            # calculate center distance between i and j box
            dist = (x1[i] - x1[j]) ** 2 + (y1[i] - y1[j]) ** 2

            # ovr = inter / areas[j]
            if dist <= thresh:
                suppressed[j] = 1
    return keep


# min_radius之内的只保留一个，最多保留post_max_size个中心点
def _circle_nms(boxes, min_radius, post_max_size=83):
    """
    NMS according to center distance
    """
    keep = np.array(circle_nms(boxes.detach().cpu().numpy(), thresh=min_radius))[:post_max_size]

    keep = torch.from_numpy(keep).long().to(boxes.device)

    return keep


# 输入值最大的前K个位置【bs 500 1】
def _gather_feat(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


def _topk(scores, K=40):
    batch, num_class, height, width = scores.size()  # 输入的heatmap 【8 1 200 176】

    topk_scores, topk_inds = torch.topk(scores.flatten(2, 3), K)  # 把heatmap展平 取前500个最大位置

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds // width).float()
    topk_xs = (topk_inds % width).int().float()

    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_classes = (topk_ind / K).int()
    topk_inds = _gather_feat(topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_classes, topk_ys, topk_xs


def decode_bbox_from_heatmap(heatmap, rot_cos, rot_sin, center, center_z, dim,
                             point_cloud_range=None, voxel_size=None, feature_map_stride=None, vel=None, K=100,
                             circle_nms=False, score_thresh=None, post_center_limit_range=None):
    batch_size, num_class, _, _ = heatmap.size()

    if circle_nms:
        # TODO: not checked yet
        assert False, 'not checked yet'
        heatmap = _nms(heatmap)

    scores, inds, class_ids, ys, xs = _topk(heatmap, K=K)  # 在heatmap上找值最大的K个位置
    # 这K个位置对应的5个预测值
    center = _transpose_and_gather_feat(center, inds).view(batch_size, K, 2)
    rot_sin = _transpose_and_gather_feat(rot_sin, inds).view(batch_size, K, 1)
    rot_cos = _transpose_and_gather_feat(rot_cos, inds).view(batch_size, K, 1)
    center_z = _transpose_and_gather_feat(center_z, inds).view(batch_size, K, 1)
    dim = _transpose_and_gather_feat(dim, inds).view(batch_size, K, 3)
    torch.clamp_(dim, 0.5, 5.5)  # 限制heatmap预测值不要太离谱

    angle = torch.atan2(rot_sin, rot_cos)
    xs = xs.view(batch_size, K, 1) + center[:, :, 0:1]  # center代表的是当前热图像素距离中心点的位置
    ys = ys.view(batch_size, K, 1) + center[:, :, 1:2]

    xs = xs * feature_map_stride * voxel_size[0] + point_cloud_range[0]  # 从feature map坐标映射回实际场景坐标
    ys = ys * feature_map_stride * voxel_size[1] + point_cloud_range[1]

    box_part_list = [xs, ys, center_z, dim, angle]
    if vel is not None:
        vel = _transpose_and_gather_feat(vel, inds).view(batch_size, K, 2)
        box_part_list.append(vel)

    final_box_preds = torch.cat((box_part_list), dim=-1)  # [bs 500 7]
    final_scores = scores.view(batch_size, K)
    final_class_ids = class_ids.view(batch_size, K)

    # 从这500个检测结果里滤除一些
    # 1.限制中心点在场景里的一定范围之内 这里其实滤除的不多
    assert post_center_limit_range is not None
    mask = (final_box_preds[..., :3] >= post_center_limit_range[:3]).all(2)
    mask &= (final_box_preds[..., :3] <= post_center_limit_range[3:]).all(2)

    # 2.滤除score_thresh过低的点 这里是0.1 这里滤了很多
    if score_thresh is not None:
        mask &= (final_scores > score_thresh)

    ret_pred_dicts = []
    for k in range(batch_size):
        cur_mask = mask[k]
        cur_boxes = final_box_preds[k, cur_mask]
        cur_scores = final_scores[k, cur_mask]
        cur_labels = final_class_ids[k, cur_mask]

        if circle_nms:
            assert False, 'not checked yet'
            centers = cur_boxes[:, [0, 1]]
            boxes = torch.cat((centers, scores.view(-1, 1)), dim=1)
            keep = _circle_nms(boxes, min_radius=min_radius, post_max_size=nms_post_max_size)

            cur_boxes = cur_boxes[keep]
            cur_scores = cur_scores[keep]
            cur_labels = cur_labels[keep]

        ret_pred_dicts.append({
            'pred_boxes': cur_boxes,
            'pred_scores': cur_scores,
            'pred_labels': cur_labels
        })
    return ret_pred_dicts  # 输出每个batch的预测结果 例如pred boxes【247， 7】


# def decode_centers_from_heatmap(heatmap, center, center_z, point_cloud_range=None, voxel_size=None, feature_map_stride=None,
#                                 K=100, circle_nms=False, score_thresh=None, post_center_limit_range=None,
#                                 min_radius=100, post_max_size=40):
#     batch_size, num_class, _, _ = heatmap.size()
#     scores, inds, class_ids, ys, xs = _topk(heatmap, K=K)  # 在heatmap上找值最大的K个位置
#
#     center = _transpose_and_gather_feat(center, inds).view(batch_size, K, 2)
#     center_z = _transpose_and_gather_feat(center_z, inds).view(batch_size, K, 1)
#     xs = xs.view(batch_size, K, 1) + center[:, :, 0:1]  # center代表的是当前热图像素距离中心点的位置
#     ys = ys.view(batch_size, K, 1) + center[:, :, 1:2]
#     xs = xs * feature_map_stride * voxel_size[0] + point_cloud_range[0]  # 从feature map坐标映射回实际场景坐标
#     ys = ys * feature_map_stride * voxel_size[1] + point_cloud_range[1]
#     final_center_preds = torch.cat(([xs, ys, center_z]), dim=-1)  # [bs 500 2]
#     final_scores = scores.view(batch_size, K)
#
#     # 从这500个检测结果里滤除一些
#     # 1.限制中心点在场景里的一定范围之内 这里其实滤除的不多
#     assert post_center_limit_range is not None
#     mask = (final_center_preds[..., :3] >= post_center_limit_range[:3]).all(2)
#     mask &= (final_center_preds[..., :3] <= post_center_limit_range[3:6]).all(2)
#
#     # 2.滤除score_thresh过低的点 这里是0.1 这里滤了很多
#     if score_thresh is not None:
#         mask &= (final_scores > score_thresh)
#
#     ret_pred_dicts = []
#     for k in range(batch_size):
#         cur_mask = mask[k]
#         cur_centers = final_center_preds[k, cur_mask]
#         cur_scores = final_scores[k, cur_mask]
#
#         if circle_nms:
#             centers = cur_centers[:, :2]
#             scores = cur_scores
#             boxes = torch.cat((centers, scores.view(-1, 1)), dim=1)
#             keep = _circle_nms(boxes, min_radius=min_radius, post_max_size=post_max_size)
#
#             cur_centers = cur_centers[keep]
#             cur_scores = cur_scores[keep]
#
#         ret_pred_dicts.append({
#             'pred_centers': cur_centers,
#             'pred_scores': cur_scores,
#         })
#     return ret_pred_dicts
