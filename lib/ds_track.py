# vim: expandtab:ts=4:sw=4
from __future__ import division, print_function, absolute_import

import argparse
import os

import cv2
import torch.nn.functional as F
import torch
import numpy as np

def box_xyxy_to_xywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [x0, y0,
         x1-x0, y1-y0]
    return torch.stack(b, dim=-1)

def get_sequence(entry, gt_annotation, shape, task="sgcls"):

    if task == "predcls":
        pass

    if task == "sgdet" or task=='sgcls':    
        
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
    
   

