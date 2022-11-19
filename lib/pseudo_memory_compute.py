import numpy as np
np.set_printoptions(precision=4)
import copy
import torch
import datetime
import time

from lib.object_detector import detector
from lib.sttran import STTran
from lib.ds_track import get_sequence

def pseudo_memory_computation(dataset,dataloader,model,object_detector,device,conf):
    rel_class_num = {'attention':model.attention_class_num,
                     'spatial': model.spatial_class_num,
                     'contacting': model.contact_class_num} 
    rel_norm_factor = {}
    rel_memory = {}
    for rel in rel_class_num.keys():
        rel_norm_factor[rel] = torch.zeros(rel_class_num[rel]).to(device)
        rel_memory[rel] = torch.zeros(rel_class_num[rel],1936).to(device)

    with torch.no_grad():
        for b, data in enumerate(dataloader): 

            print('video-> ',data[4], flush=True)
            im_data = copy.deepcopy(data[0].to(device))
            im_info = copy.deepcopy(data[1].to(device))
            gt_boxes = copy.deepcopy(data[2].to(device))
            num_boxes = copy.deepcopy(data[3].to(device))
            gt_annotation = dataset.gt_annotations[data[4]]

            entry = object_detector(im_data, im_info, gt_boxes, num_boxes, gt_annotation, im_all=None)

            if conf.tracking:
                get_sequence(entry, gt_annotation, (im_info[0][:2]/im_info[0,2]).cpu().data,conf.mode)
            
            pred = model(entry,phase='test', unc=False)

            rel_features = pred['rel_features']
            thresh = {'attention':1,'contacting':0,'spatial':0}
            thresh['spatial'] = (conf.pseudo_thresh-1)//2
            if  thresh['spatial'] >  rel_class_num['spatial']:
                thresh['spatial'] = rel_class_num['spatial']

            thresh['contacting'] = (conf.pseudo_thresh-1-thresh['spatial'])
            if thresh['contacting'] >  rel_class_num['contacting']:
                thresh['contacting'] = rel_class_num['contacting']

            for rel in rel_class_num.keys():
                rel_pseudo_labels = torch.topk(pred[rel+"_distribution"], thresh[rel], dim=-1, largest=True, sorted=True)[1]

                pseudo_one_hot = torch.zeros(pred[rel+"_distribution"].shape).to(device) # N x C
                pseudo_one_hot = pseudo_one_hot.scatter_(1, rel_pseudo_labels, 1.)
                
                rel_memory[rel] = rel_memory[rel] + torch.matmul(pseudo_one_hot.T,rel_features)
                for c in range(pseudo_one_hot.shape[1]):
                    rel_norm_factor[rel][c] += pseudo_one_hot[:,c].sum(0)
        
        for rel in rel_memory.keys():
            tmp = rel_memory[rel]
            nz_idx = torch.where(rel_norm_factor[rel]!=0) 
            tmp[nz_idx] = tmp[nz_idx]/(rel_norm_factor[rel][nz_idx].unsqueeze(-1).repeat(1,1936))
            rel_memory[rel] = tmp
    
    model.rel_memory = rel_memory
                



            # pseudo_labels['attention'] = torch.argmax(pred["attention_distribution"])
            # spa_pseudo_labels = torch.topk(pred["spatial_distribution"], spa_thresh, dim=-1, largest=True, sorted=True)[1]
            # con_pseudo_labels = torch.topk(pred["contacting_distribution"], con_thresh, dim=-1, largest=True, sorted=True)[1]

            