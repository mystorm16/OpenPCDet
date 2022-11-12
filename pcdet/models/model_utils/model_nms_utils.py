import torch

from ...ops.iou3d_nms import iou3d_nms_utils


def class_agnostic_nms(box_scores, box_preds, nms_config, score_thresh=None):
    # 1.首先根据置信度阈值过滤掉部过滤掉大部分置信度低的box，加速后面的nms操作
    src_box_scores = box_scores
    if score_thresh is not None:
        # 得到类别预测概率大于score_thresh的mask
        scores_mask = (box_scores >= score_thresh)
        box_scores = box_scores[scores_mask]
        box_preds = box_preds[scores_mask]

    # 初始化空列表，用来存放经过nms后保留下来的anchor
    selected = []
    # 如果有anchor的类别预测大于score_thresh的话才进行nms，否则返回空
    if box_scores.shape[0] > 0:
        # 这里只保留最大的K个anchor置信度来进行nms操作
        # k取min(nms_config.NMS_PRE_MAXSIZE, box_scores.shape[0])的最小值
        box_scores_nms, indices = torch.topk(box_scores, k=min(nms_config.NMS_PRE_MAXSIZE, box_scores.shape[0]))
        # box_scores_nms只是得到了类别的更新结果；
        # 此处更新box的预测结果 根据tokK重新选取并从大到小排序的结果 更新boxes的预测
        boxes_for_nms = box_preds[indices]
        # 调用iou3d_nms_utils的nms_gpu函数进行nms，
        # 返回的是被保留下的box的索引，selected_scores = None
        # 根据返回索引找出box索引值
        keep_idx, selected_scores = getattr(iou3d_nms_utils, nms_config.NMS_TYPE)(
                boxes_for_nms[:, 0:7], box_scores_nms, nms_config.NMS_THRESH, **nms_config
        )
        selected = indices[keep_idx[:nms_config.NMS_POST_MAXSIZE]]

    if score_thresh is not None:
        original_idxs = scores_mask.nonzero().view(-1)
        selected = original_idxs[selected]
    return selected, src_box_scores[selected]


def multi_classes_nms(cls_scores, box_preds, nms_config, score_thresh=None):
    """
    Args:
        cls_scores: (N, num_class)
        box_preds: (N, 7 + C)
        nms_config:
        score_thresh:

    Returns:

    """
    pred_scores, pred_labels, pred_boxes = [], [], []
    for k in range(cls_scores.shape[1]):
        if score_thresh is not None:
            scores_mask = (cls_scores[:, k] >= score_thresh)
            box_scores = cls_scores[scores_mask, k]
            cur_box_preds = box_preds[scores_mask]
        else:
            box_scores = cls_scores[:, k]
            cur_box_preds = box_preds

        selected = []
        if box_scores.shape[0] > 0:
            box_scores_nms, indices = torch.topk(box_scores, k=min(nms_config.NMS_PRE_MAXSIZE, box_scores.shape[0]))
            boxes_for_nms = cur_box_preds[indices]
            keep_idx, selected_scores = getattr(iou3d_nms_utils, nms_config.NMS_TYPE)(
                    boxes_for_nms[:, 0:7], box_scores_nms, nms_config.NMS_THRESH, **nms_config
            )
            selected = indices[keep_idx[:nms_config.NMS_POST_MAXSIZE]]

        pred_scores.append(box_scores[selected])
        pred_labels.append(box_scores.new_ones(len(selected)).long() * k)
        pred_boxes.append(cur_box_preds[selected])

    pred_scores = torch.cat(pred_scores, dim=0)
    pred_labels = torch.cat(pred_labels, dim=0)
    pred_boxes = torch.cat(pred_boxes, dim=0)

    return pred_scores, pred_labels, pred_boxes
