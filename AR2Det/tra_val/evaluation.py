from __future__ import division
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from math import pi
from collections import defaultdict
import itertools
import numpy as np
import six
import torch
import cv2
import tensorflow as tf
from box_utils.rotate_polygon_nms import rotate_gpu_nms

#using for predict IOU, in this paper, it is not be used.
def bbox_iou_scores1_1(boxes1, boxes2, pre_regions, size=[128,128],use_gpu=True):
    area1 = boxes1[:, :, 2] * boxes1[:, :, 3]
    area2 = boxes2[:, :, 2] * boxes2[:, :, 3]
    if use_gpu:
        ious = torch.zeros(size[0],size[1]).cuda()
    else:
        ious = torch.zeros(size[0],size[1])
    for i in range(size[0]):
        if torch.sum(pre_regions[i,:]) == 0:
            continue
        for j in range(size[1]):
            if  pre_regions[i,j]==1:
                r1 = ((boxes1[i,j,0], boxes1[i,j,1]), (boxes1[i,j,2], boxes1[i,j,3]), (boxes1[i,j,4]/pi)*180)
                r2 = ((boxes2[i,j,0], boxes2[i,j,1]), (boxes2[i,j,2], boxes2[i,j,3]), (boxes2[i,j,4]/pi)*180)
                int_pts = cv2.rotatedRectangleIntersection(r1, r2)[1]
                if int_pts is not None:
                    order_pts = cv2.convexHull(int_pts, returnPoints=True)
                    int_area = cv2.contourArea(order_pts)
                    inter = int_area * 1.0 / (area1[i,j] + area2[i,j] - int_area)
                    ious[i,j] = inter
                else:
                    ious[i,j] = 0.0
    return ious.detach()

def bbox_iou_scores1_n(boxes1, boxes2, pre_re, size=[128,128],use_gpu=True):
    area1 = boxes1[:, :, 2] * boxes1[:, :, 3]
    area2 = boxes2[:, 2] * boxes2[:, 3]
    if use_gpu:
        ious = torch.zeros(size[0],size[1]).cuda()
    else:
        ious = torch.zeros(size[0],size[1])
    gb_num = boxes2.shape[0]
    for i in range(size[0]):
        for j in range(size[1]):
            if pre_re[i,j] ==1:
                max_iou = 0
                r1 = ((boxes1[i,j,0], boxes1[i,j,1]), (boxes1[i,j,2], boxes1[i,j,3]), (boxes1[i,j,4]/pi)*180)
                for gb in range(gb_num):
                    boxes2_ = boxes2[gb]
                    r2 = ((boxes2_[0], boxes2_[1]), (boxes2_[2], boxes2_[3]), (boxes2_[4]/pi)*180)

                    int_pts = cv2.rotatedRectangleIntersection(r1, r2)[1]
                    if int_pts is not None:
                        order_pts = cv2.convexHull(int_pts, returnPoints=True)
                        int_area = cv2.contourArea(order_pts)
                        inter = int_area * 1.0 / (area1[i,j] + area2[gb] - int_area)
                    else:
                        inter = 0.0
                    if inter > max_iou:
                        max_iou = inter
                ious[i,j] = max_iou
    return ious.detach()

def eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels,
        gt_difficults=None,
        iou_thresh=0.5, use_07_metric=False):
    
    prec, rec = calc_detection_voc_prec_rec(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels, gt_difficults,
        iou_thresh=iou_thresh)

    ap,prec,rec = calc_detection_voc_ap(prec, rec, use_07_metric=use_07_metric)

    return {'ap': ap, 'map': np.nanmean(ap)},prec, rec

def calc_detection_voc_prec_rec(
        pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels,
        gt_difficults=None,
        iou_thresh=0.7):

    pred_bboxes = iter(pred_bboxes)
    pred_labels = iter(pred_labels)
    pred_scores = iter(pred_scores)
    gt_bboxes = iter(gt_bboxes)
    gt_labels = iter(gt_labels)
    if gt_difficults is None:
        gt_difficults = itertools.repeat(None)
    else:
        gt_difficults = iter(gt_difficults)

    n_pos = defaultdict(int)
    score = defaultdict(list)
    match = defaultdict(list)

    
    for pred_bbox, pred_label, pred_score, gt_bbox, gt_label, gt_difficult in \
            six.moves.zip(
                pred_bboxes, pred_labels, pred_scores,
                gt_bboxes, gt_labels, gt_difficults):
        

        if gt_difficult is None:
            gt_difficult = np.zeros(gt_bbox.shape[0], dtype=bool)

        for l in np.unique(np.concatenate((pred_label, gt_label)).astype(int)):
            pred_mask_l = pred_label == l
            pred_bbox_l = pred_bbox[pred_mask_l]
            pred_score_l = pred_score[pred_mask_l]
            # sort by score
            order = pred_score_l.argsort()[::-1]
            pred_bbox_l = pred_bbox_l[order]
            pred_score_l = pred_score_l[order]

            gt_mask_l = gt_label == l
            gt_bbox_l = gt_bbox[gt_mask_l]
            gt_difficult_l = gt_difficult[gt_mask_l]

            n_pos[l] += np.logical_not(gt_difficult_l).sum()
            score[l].extend(pred_score_l)

            if len(pred_bbox_l) == 0:
                continue
            if len(gt_bbox_l) == 0:
                match[l].extend((0,) * pred_bbox_l.shape[0])
                continue

            pred_bbox_l = pred_bbox_l.copy()
            gt_bbox_l = gt_bbox_l.copy()

            iou = bbox_iou(pred_bbox_l, gt_bbox_l)
            iou = np.stack(iou)
            gt_index = iou.argmax(axis=1)
            # set -1 if there is no matching ground truth
            gt_index[iou.max(axis=1) < iou_thresh] = -1
            del iou

            selec = np.zeros(gt_bbox_l.shape[0], dtype=bool)
            for gt_idx in gt_index:
                if gt_idx >= 0:
                    if gt_difficult_l[gt_idx]:
                        match[l].append(-1)
                    else:
                        if not selec[gt_idx]:
                            match[l].append(1)
                        else:
                            match[l].append(0)
                    selec[gt_idx] = True
                else:
                    match[l].append(0)

    for iter_ in (
            pred_bboxes, pred_labels, pred_scores,
            gt_bboxes, gt_labels, gt_difficults):
        if next(iter_, None) is not None:
            raise ValueError('Length of input iterables need to be same.')

    n_fg_class = max(n_pos.keys()) + 1
    prec = [None] * n_fg_class
    rec = [None] * n_fg_class

    for l in n_pos.keys():
        score_l = np.array(score[l])
        match_l = np.array(match[l], dtype=np.int8)

        order = score_l.argsort()[::-1]
        match_l = match_l[order]

        tp = np.cumsum(match_l == 1)
        fp = np.cumsum(match_l == 0)

        # If an element of fp + tp is 0,
        # the corresponding element of prec[l] is nan.
        prec[l] = tp / (fp + tp)
        # If n_pos[l] is 0, rec[l] is None.
        if n_pos[l] > 0:
            rec[l] = tp / n_pos[l]
    return prec, rec

def calc_detection_voc_ap(prec, rec, use_07_metric=False):
    n_fg_class = len(prec)
    ap = np.empty(n_fg_class)
    for l in six.moves.range(n_fg_class):
        if prec[l] is None or rec[l] is None:
            ap[l] = np.nan
            continue

        if use_07_metric:
            # 11 point metric
            ap[l] = 0
            for t in np.arange(0., 1.1, 0.1):
                if np.sum(rec[l] >= t) == 0:
                    p = 0
                else:
                    p = np.max(np.nan_to_num(prec[l])[rec[l] >= t])
                ap[l] += p / 11
        else:
            # correct AP calculation
            # first append sentinel values at the end
            mpre = np.concatenate(([0], np.nan_to_num(prec[l]), [0]))
            mrec = np.concatenate(([0], rec[l], [1]))

            mpre = np.maximum.accumulate(mpre[::-1])[::-1]

            # to calculate area under PR curve, look for points
            # where X axis (recall) changes value
            i = np.where(mrec[1:] != mrec[:-1])[0]

            # and sum (\Delta recall) * prec
            ap[l] = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap, prec, rec

# -*- coding: utf-8 -*-


def nms_rotate(decode_boxes, scores, iou_threshold, max_output_size,
               use_angle_condition=True, angle_threshold=0, use_gpu=True, gpu_id=0):
    """
    :param boxes: format [x_c, y_c, w, h, theta]
    :param scores: scores of boxes
    :param threshold: iou threshold (0.7 or 0.5)
    :param max_output_size: max number of output
    :return: the remaining index of boxes
    """

    if use_gpu:
        keep = nms_rotate_gpu(boxes_list=decode_boxes,
                              scores=scores,
                              iou_threshold=iou_threshold,
                              angle_gap_threshold=angle_threshold,
                              use_angle_condition=use_angle_condition,
                              device_id=gpu_id)

        keep = tf.cond(
            tf.greater(tf.shape(keep)[0], max_output_size),
            true_fn=lambda: tf.slice(keep, [0], [max_output_size]),
            false_fn=lambda: keep)

    else:
        keep = tf.py_func(nms_rotate_cpu,
                          inp=[decode_boxes, scores, iou_threshold, max_output_size],
                          Tout=tf.int64)
    return keep


def nms_rotate_cpu(boxes, scores, iou_threshold, max_output_size):
    print(scores,len(scores))
    keep = []

    order = scores.argsort()[::-1]
    print(order,len(order))
    num = boxes.shape[0]

    suppressed = np.zeros((num), dtype=np.int)

    for _i in range(num):
        if len(keep) >= max_output_size:
            break

        i = order[_i]
        if suppressed[i] == 1:
            continue
        keep.append(i)
        r1 = ((boxes[i, 0], boxes[i, 1]), (boxes[i, 2], boxes[i, 3]), boxes[i, 4])
        area_r1 = boxes[i, 2] * boxes[i, 3]
        for _j in range(_i + 1, num):
            j = order[_j]
            if suppressed[i] == 1:
                continue
            r2 = ((boxes[j, 0], boxes[j, 1]), (boxes[j, 2], boxes[j, 3]), boxes[j, 4])
            area_r2 = boxes[j, 2] * boxes[j, 3]
            inter = 0.0

            try:
                int_pts = cv2.rotatedRectangleIntersection(r1, r2)[1]

                if int_pts is not None:
                    order_pts = cv2.convexHull(int_pts, returnPoints=True)

                    int_area = cv2.contourArea(order_pts)

                    inter = int_area * 1.0 / (area_r1 + area_r2 - int_area + 1e-5)

            except:
                """
                  cv2.error: /io/opencv/modules/imgproc/src/intersection.cpp:247:
                  error: (-215) intersection.size() <= 8 in function rotatedRectangleIntersection
                """
                # print(r1)
                # print(r2)
                inter = 0.9999

            if inter >= iou_threshold:
                suppressed[j] = 1

    return np.array(keep, np.int64)


def nms_rotate_gpu(boxes_list, scores, iou_threshold, use_angle_condition=True, angle_gap_threshold=0, device_id=0):
    
    if use_angle_condition:
        x_c, y_c, w, h, theta = tf.unstack(boxes_list, axis=1)
        boxes_list = tf.transpose(tf.stack([x_c, y_c, w, h, theta]))
        det_tensor = tf.concat([boxes_list, tf.expand_dims(scores, axis=1)], axis=1)
        keep = tf.py_func(rotate_gpu_nms,
                          inp=[det_tensor, iou_threshold, device_id],
                          Tout=tf.int64)
        return keep
    else:
        x_c, y_c, w, h, theta = tf.unstack(boxes_list, axis=1)
        boxes_list = tf.transpose(tf.stack([x_c, y_c, w, h, theta]))
        det_tensor = tf.concat([boxes_list, tf.expand_dims(scores, axis=1)], axis=1)
        keep = tf.py_func(rotate_gpu_nms,
                          inp=[det_tensor, iou_threshold, device_id],
                          Tout=tf.int64)
        keep = tf.reshape(keep, [-1])
        return keep
    
def iou_rotate_calculate(boxes1, boxes2, use_gpu=True, gpu_id=1):

    area1 = boxes1[:, 2] * boxes1[:, 3]
    area2 = boxes2[:, 2] * boxes2[:, 3]
    ious = []
    for i, box1 in enumerate(boxes1):
        temp_ious = []
        r1 = ((box1[0], box1[1]), (box1[2], box1[3]), (box1[4]/pi)*180)
        for j, box2 in enumerate(boxes2):
            r2 = ((box2[0], box2[1]), (box2[2], box2[3]), (box2[4]/pi)*180)
            int_pts = cv2.rotatedRectangleIntersection(r1, r2)[1]
            if int_pts is not None:
                order_pts = cv2.convexHull(int_pts, returnPoints=True)

                int_area = cv2.contourArea(order_pts)

                inter = int_area * 1.0 / (area1[i] + area2[j] - int_area)
                temp_ious.append(inter)
            else:
                temp_ious.append(0.0)
        ious.append(temp_ious)
    return ious

def bbox_iou(bbox_a, bbox_b):
    if bbox_a.shape[1] == 5 and bbox_b.shape[1] == 5:
        return iou_rotate_calculate(bbox_a, bbox_b)

