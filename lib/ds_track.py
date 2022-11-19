# vim: expandtab:ts=4:sw=4
from __future__ import division, print_function, absolute_import

import argparse
import os

import cv2
import torch.nn.functional as F
import torch
import numpy as np
# from fasterRCNN.lib.model.roi_layers import ROIAlign, nms
# from lib.deep_sort.application_util import preprocessing,visualization
# from lib.deep_sort.deep_sort import nn_matching
# from lib.deep_sort.deep_sort.detection import Detection
# from lib.deep_sort.deep_sort.tracker import Tracker




# def create_detections(final_boxes,final_features,final_conf, frame_idx, min_height=0):
#     """Create detections for given frame index from the raw detection matrix.

#     Parameters
#     ----------
#     detection_mat : ndarray
#         Matrix of detections. The first 10 columns of the detection matrix are
#         in the standard MOTChallenge detection format. In the remaining columns
#         store the feature vector associated with each detection.
#     frame_idx : int
#         The frame index.
#     min_height : Optional[int]
#         A minimum detection bounding box height. Detections that are smaller
#         than this value are disregarded.

#     Returns
#     -------
#     List[tracker.Detection]
#         Returns detection responses at given frame index.

#     """
#     frame_indices = final_boxes[:, 0].astype(np.int)
#     mask = frame_indices == frame_idx

#     detection_list = []
#     for b,c,f in zip(final_boxes[mask],final_conf[mask],final_features[mask]):
#         bbox, confidence, feature = b[1:], c,f
#         if bbox[3] < min_height:
#             continue
#         detection_list.append(Detection(bbox, confidence, feature))
#     return detection_list


# def run(final_boxes,final_features,final_conf, min_confidence=0.3,
#         nms_max_overlap=1, min_detection_height=0, max_cosine_distance=0.2,
#         nn_budget=100, display=False):
#     """Run multi-target tracker on a particular sequence.

#     Parameters
#     ----------
   
#     min_confidence : float
#         Detection confidence threshold. Disregard all detections that have
#         a confidence lower than this value.
#     nms_max_overlap: float
#         Maximum detection overlap (non-maxima suppression threshold).
#     min_detection_height : int
#         Detection height threshold. Disregard all detections that have
#         a height lower than this value.
#     max_cosine_distance : float
#         Gating threshold for cosine distance metric (object appearance).
#     nn_budget : Optional[int]
#         Maximum size of the appearance descriptor gallery. If None, no budget
#         is enforced.
#     display : bool
#         If True, show visualization of intermediate tracking results.

#     """
#     seq_info = {'min_frame_idx':final_boxes[0,0],
#                 'max_frame_idx':final_boxes[-1,0]}
   
#     metric = nn_matching.NearestNeighborDistanceMetric(
#         "cosine", max_cosine_distance, nn_budget)
#     tracker = Tracker(metric,max_iou_distance=0.7,n_init=1)
#     results = []

#     def frame_callback(vis, frame_idx):
#         # print("Processing frame %05d" % frame_idx)

#         # Load image and generate detections.
#         detections = create_detections(
#             final_boxes,final_features,final_conf, frame_idx, min_detection_height)
#         detections = [d for d in detections if d.confidence >= min_confidence]

#         # Run non-maxima suppression.
#         boxes = np.array([d.tlwh for d in detections])
#         scores = np.array([d.confidence for d in detections])
#         ############## nms already done #######
#         # indices = preprocessing.non_max_suppression(
#         #     boxes, nms_max_overlap, scores)
#         # print(indices)
#         # print(60*'x')
#         indices = np.argsort(scores)
#         # print(indices)
#         # exit()
#         detections = [detections[i] for i in indices]
        
#         # Update tracker.
#         tracker.predict()
#         tracker.update(detections)

#         # Update visualization.
#         if display:
#             image = cv2.imread(
#                 seq_info["image_filenames"][frame_idx], cv2.IMREAD_COLOR)
#             vis.set_image(image.copy())
#             vis.draw_detections(detections)
#             vis.draw_trackers(tracker.tracks)

#         # Store results.
#         for track in tracker.tracks:
#             if not track.is_confirmed() or track.time_since_update > 1:
#                 continue
#             bbox = track.to_tlwh()
#             results.append([
#                 frame_idx, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3]])

#     # Run tracker.
#     if display:
#         visualizer = visualization.Visualization(seq_info, update_ms=5)
#     else:
#         visualizer = visualization.NoVisualization(seq_info)
#     visualizer.run(frame_callback)
#     results = np.asarray(results)
#     l=[]
#     # print(results)
#     print(100*'===')
#     for ll in results:
#         l.append(ll[1])
#     u, i = np.unique(l, return_counts=True)
#     print(u,i)
#     print(len(results),len(final_boxes))
#     # print(u[np.bincount(i) > 1])
#     return results
#     # Store results.
#     # f = open(output_file, 'w')
#     # for row in results:
#     #     print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (
#     #         row[0], row[1], row[2], row[3], row[4], row[5]),file=f)


# def bool_string(input_string):
#     if input_string not in {"True","False"}:
#         raise ValueError("Please Enter a valid Ture/False choice")
#     else:
#         return (input_string == "True")

# def clean_bbox(entry):
#     # nms for clustering
#     final_boxes = []
#     final_feats = []
#     final_dists = []
#     final_labels = []
#     box_counts = 0
#     counts = 0
#     mapping = {}
#     for i in range(int(entry["boxes"][-1,0])):
#         # images in the batch
#         scores = entry['distribution'][entry['boxes'][:, 0] == i]
#         pred_boxes = entry['boxes'][entry['boxes'][:, 0] == i, 1:]
#         feats = entry['features'][entry['boxes'][:, 0] == i]
#         labels = entry['labels'][entry['boxes'][:, 0] == i]
        
#         for j in torch.argmax(scores, dim=1).unique():
#             # NMS according to obj categories
#             inds = torch.nonzero(torch.argmax(scores, dim=1) == j).view(-1)
#             # if there is det
#             if inds.numel() > 0:
#                 cls_dists = scores[inds]
#                 cls_feats = feats[inds]
#                 cls_dists = scores[inds]
#                 cls_labels = labels[inds]
#                 cls_scores = cls_dists[:, j]
#                 _, order = torch.sort(cls_scores, 0, True)            
#                 cls_boxes = pred_boxes[inds]
#                 cls_dists = cls_dists[order]
#                 cls_feats = cls_feats[order]
#                 cls_labels = cls_labels[order]
#                 keep = nms(cls_boxes[order, :], cls_scores[order], 0.4) # hyperparameter
#                 not_keep = torch.LongTensor([k for k in range(len(inds)) if k not in keep])
#                 if len(not_keep) > 0:
#                     anchor = cls_boxes[keep][:,0:]
#                     remain = cls_boxes[not_keep][:,0:]
#                     # alignment = torch.argmax(generalized_box_iou(anchor, remain),0)
#                 else:
#                     alignment = []
#                 final_dists.append(cls_dists[keep.view(-1).long()])
#                 final_boxes.append(torch.cat((torch.tensor([[i]], dtype=torch.float).repeat(keep.shape[0],
#                                                                                             1).cuda(0),
#                                               cls_boxes[order, :][keep.view(-1).long()]), 1))
#                 final_feats.append(cls_feats[keep.view(-1).long()])
#                 final_labels.append(cls_labels[keep.view(-1).long()])
#                 for k, ind in enumerate(keep):
#                     key = counts + k
#                     value = inds[order[ind]] + box_counts
#                     mapping[key] = [value.item()]
#                 # for ind, align in zip(not_keep, alignment):
#                 #     key = counts + align
#                 #     value = inds[order[ind]] + box_counts
#                 #     mapping[key.item()].append(value.item())
#                 counts += len(keep)
#         box_counts += len(pred_boxes)
#         """
#         # ignore predicted classes
#         cls_scores, _ = torch.max(scores,1)
#         _, order = torch.sort(cls_scores, 0, True)
#         cls_boxes = pred_boxes
#         cls_dists = scores[order]
#         cls_feats = feats[order]
#         cls_labels = labels[order]
#         keep =  nms(cls_boxes[order, :], cls_scores[order], 0.5) # hyperparameter
#         not_keep = torch.LongTensor([k for k in range(len(order)) if k not in keep])
#         if len(not_keep) > 0:
#             anchor = cls_boxes[keep][:,0:]
#             remain = cls_boxes[not_keep][:,0:]
#             alignment = torch.argmax(generalized_box_iou(anchor, remain),0)
#         else:
#             alignment = []
#         final_dists.append(cls_dists[keep.view(-1).long()])
#         final_boxes.append(torch.cat((torch.tensor([[i]], dtype=torch.float).repeat(keep.shape[0],
#                                                                                     1).cuda(0),
#                                       cls_boxes[order, :][keep.view(-1).long()]), 1))
#         final_feats.append(cls_feats[keep.view(-1).long()])
#         final_labels.append(cls_labels[keep.view(-1).long()])
#         for k, ind in enumerate(keep):
#             key = counts + k
#             value = order[ind] + box_counts
#             mapping[key] = [value.item()]
#         for ind, align in zip(not_keep, alignment):
#             key = counts + align
#             value = order[ind] + box_counts
#             mapping[key.item()].append(value.item())
#         counts += len(keep)   
#         box_counts += len(pred_boxes)
#         """
#     final_boxes = torch.cat(final_boxes,0)
#     final_dists = torch.cat(final_dists, dim=0)
#     final_feats = torch.cat(final_feats, 0)
#     final_labels = torch.cat(final_labels, 0)
#     print(entry['boxes'].shape)
#     print(final_boxes.shape)
#     return final_boxes, final_feats, final_dists, final_labels, mapping

def box_xyxy_to_xywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [x0, y0,
         x1-x0, y1-y0]
    return torch.stack(b, dim=-1)

def get_sequence(entry, gt_annotation, shape, task="sgcls"):

    if task == "predcls":
        pass

    if task == "sgdet" or task=='sgcls':    
        # for sgdet, use the predicted object classes, as a special case of 
        # the proposed method, comment this out for general coase tracking.
        indices = [[]]
        # indices[0] store single-element sequence, to save memory
        pred_labels = torch.argmax(entry["distribution"], 1)
        for i in pred_labels.unique():
            index = torch.where(pred_labels==i)[0]
            if len(index) == 1:
                indices[0].append(index)
            else:
                indices.append(index)
        if len(indices[0])>0:
            indices[0] = torch.cat(indices[0])
        else:
            indices[0] = torch.tensor([])
        entry["indices"] = indices         
        return 
    
    # w, h = shape
    # key_frames = np.array([annotation[0]["frame"] for annotation in gt_annotation])
    # cluster = []
    # cluster_feature = []
    # cluster_dist = []
    # last_key = -100
    # tracks = []

    # if task == "sgdet":
    #     final_boxes, final_features, final_dists, final_labels, mapping = clean_bbox(entry)
    #     final_boxes[:,1:] = box_xyxy_to_xywh(final_boxes[:,1:])
    #     final_pred = final_dists.argmax(1)
    #     final_dists = F.one_hot(final_pred, final_dists.shape[1]).float()
    # elif task == "sgcls":
    #     final_boxes = entry["boxes"]
    #     final_boxes[:,1:] = box_xyxy_to_xywh(final_boxes[:,1:])
    #     final_features = entry["features"]   
    #     final_dists = entry["distribution"]
    #     final_pred = final_dists.argmax(1)
    #     final_dists = F.one_hot(final_pred, final_dists.shape[1]).float()
    # else:
    #     print("%s is not defined"%task)
    #     assert False

    # results = run(final_boxes.cpu().clone().numpy(),
    #              final_features.cpu().clone().numpy(),
    #              final_dists.max(-1)[0].cpu().clone().numpy())
    # return results


