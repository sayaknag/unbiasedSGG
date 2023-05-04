import torch
import os
import numpy as np
from statistics import mean, median, variance, pstdev
import copy
from lib.ds_track import get_sequence

class uncertainty_values:
    def __init__(self,obj_classes,attention_class_num,spatial_class_num,contact_class_num):
        self.unc_list_obj = {}
        self.unc_list_rel = {}

        self.obj_batch_unc = {'al':[],'ep':[]}

        self.rel_batch_unc = {'attention':{'al':[],'ep':[]},
                                'spatial':{'al':[],'ep':[]},
                                'contacting':{'al':[],'ep':[]}}

        cls_obj_uc = {}
        for cls in range( obj_classes):
            if cls not in cls_obj_uc.keys():
                cls_obj_uc[cls] = {'al':[],'ep':[]}
        self.cls_obj_uc = cls_obj_uc

        cls_rel_uc = {'attention':{},
                    'spatial':{},
                    'contacting':{}}
        for k in cls_rel_uc.keys():
            if k == 'attention':
                rel_cls_num =  attention_class_num
            elif k == 'spatial':
                rel_cls_num =  spatial_class_num
            elif k == 'contacting':
                rel_cls_num =  contact_class_num

            for cls in range(rel_cls_num):
                if cls not in cls_rel_uc[k].keys():
                    cls_rel_uc[k][cls] = {'al':[],'ep':[]}

        self.cls_rel_uc = cls_rel_uc


    def stats(self):
        for k in self.cls_obj_uc.keys():
            if len(self.cls_obj_uc[k]['al']) > 0 :
                self.cls_obj_uc[k]['al'] = (mean(self.cls_obj_uc[k]['al']),pstdev(self.cls_obj_uc[k]['al']))
                self.cls_obj_uc[k]['ep'] = (mean(self.cls_obj_uc[k]['ep']),pstdev(self.cls_obj_uc[k]['ep']))

        for rel in self.cls_rel_uc.keys():
            for cls in self.cls_rel_uc[rel].keys():
                if len(self.cls_rel_uc[rel][cls]['al']) > 0:
                    self.cls_rel_uc[rel][cls]['al'] = (mean(self.cls_rel_uc[rel][cls]['al']),pstdev(self.cls_rel_uc[rel][cls]['al']))
                    self.cls_rel_uc[rel][cls]['ep'] = (mean(self.cls_rel_uc[rel][cls]['ep']),pstdev(self.cls_rel_uc[rel][cls]['ep']))

    def stats2(self):
        for k in self.cls_obj_uc.keys():
            if len(self.cls_obj_uc[k]['al']) > 0 :
                self.cls_obj_uc[k]['both'] = sum(np.exp(self.cls_obj_uc[k]['al']+self.cls_obj_uc[k]['ep']))
                self.cls_obj_uc[k]['al'] = sum(np.exp(self.cls_obj_uc[k]['al']))
                self.cls_obj_uc[k]['ep'] = sum(np.exp(self.cls_obj_uc[k]['ep']))

        for rel in self.cls_rel_uc.keys():
            for cls in self.cls_rel_uc[rel].keys():
                if len(self.cls_rel_uc[rel][cls]['al']) > 0:
                    self.cls_rel_uc[rel][cls]['both'] = sum(np.exp(self.cls_rel_uc[rel][cls]['al']+self.cls_rel_uc[rel][cls]['al']))
                    self.cls_rel_uc[rel][cls]['al'] = sum(np.exp(self.cls_rel_uc[rel][cls]['al']))
                    self.cls_rel_uc[rel][cls]['ep'] = sum(np.exp(self.cls_rel_uc[rel][cls]['ep']))


def uncertainty_computation(data,dataset,object_detector,model,unc_vals,device,
                            output_dir,obj_mem=False,obj_unc=True,background_mem=True, rel_unc=True,
                            tracking=None):

    obj_emb_path = output_dir+'obj_embeddings/'
    rel_emb_path = output_dir+'rel_embeddings/'
    if not os.path.exists(obj_emb_path):
        os.makedirs(obj_emb_path)
    if not os.path.exists(rel_emb_path):
        os.makedirs(rel_emb_path)

    model.eval()
    # for i,data in enumerate(data_loader):
    with torch.no_grad():
        im_data = copy.deepcopy(data[0].to(device=device))
        im_info = copy.deepcopy(data[1].to(device=device))
        gt_boxes = copy.deepcopy(data[2].to(device=device))
        num_boxes = copy.deepcopy(data[3].to(device=device))
        gt_annotation = dataset.gt_annotations[data[4]]
        index = data[4]

        entry = object_detector(im_data, im_info, gt_boxes, num_boxes, gt_annotation ,im_all=None)
        if tracking:
            get_sequence(entry, gt_annotation, (im_info[0][:2]/im_info[0,2]).cpu().data,model.mode)
        pred = model(entry,unc=True)

        obj_labels = pred['labels']
        unq_labels = torch.unique(obj_labels)
        # obj_cls_al_uc = pred['obj_al_uc'][torch.arange(pred['obj_al_uc'].size(0)),obj_labels].cpu()
        # obj_cls_ep_uc = pred['obj_ep_uc'][torch.arange(pred['obj_ep_uc'].size(0)),obj_labels].cpu()
        if obj_mem:
            obj_features = pred['object_features']
            if not background_mem:
                obj_features = obj_features[obj_labels != 0 ,:]
            np.save(obj_emb_path+str(index)+'.npy',obj_features.detach().cpu().numpy(),allow_pickle=True)
            # np.save(obj_emb_path+str(index)+'_labels.npy',pred['labels'].detach().cpu().numpy(),allow_pickle=True)
        
        if obj_unc :
            
            tmp_dict = {}
            # print(pred['obj_al_uc'].shape,pred['obj_ep_uc'].shape)
            for u in ['al','ep']:
                unc_vals.obj_batch_unc[u] = pred['obj_'+u+'_uc'].cpu()

                obj_cls_uc = pred['obj_'+u+'_uc'][torch.arange(pred['obj_'+u+'_uc'].size(0)),obj_labels].cpu()

                # print(obj_cls_uc.shape)
                batch_unc = torch.zeros(obj_labels.shape[0],pred['obj_'+u+'_uc'].size(1))
                batch_unc[torch.arange(batch_unc.size(0)),obj_labels] = obj_cls_uc
                # if background memory not wanted
                if not background_mem:
                    batch_unc = batch_unc[:,1:]
                    batch_unc = batch_unc[obj_labels!=0,:]
                tmp_dict[u] = batch_unc.numpy()

                for cls in unq_labels:
                    if not background_mem and cls == 0:
                        continue
                    unc_vals.cls_obj_uc[cls.item()][u] += obj_cls_uc[torch.where(obj_labels == cls)[0]].tolist()
                    
            unc_vals.unc_list_obj[index]= tmp_dict
        else:
            tmp_dict = {}
            batch_unc = torch.zeros(obj_labels.shape[0],len(model.obj_classes))
            batch_unc[torch.arange(batch_unc.size(0)),obj_labels] = 1
            if not background_mem:
                batch_unc = batch_unc[:,1:]
                batch_unc = batch_unc[obj_labels!=0,:]
            tmp_dict['al'] = batch_unc.numpy()
    #             tmp_list = []
    #             for j in range(len(obj_labels)):
    #                 tmp_list.append({obj_labels[j].cpu().item() : (obj_cls_al_uc[j],obj_cls_ep_uc[j])})

            unc_vals.unc_list_obj[index]= tmp_dict

        
        np.save(rel_emb_path+str(index)+'.npy',pred['rel_features'].detach().cpu().numpy(),allow_pickle=True)
        rel_labels = {'attention':pred["attention_gt"],
                    'spatial':  pred["spatial_gt"],
                    'contacting': pred['contacting_gt']}

        for rel,label in rel_labels.items():
            np.save(rel_emb_path+str(index)+'_'+rel+'_labels.npy',label,allow_pickle=True)

        tmp_dict = {}
        
        for rel in rel_labels.keys():
            tmp_dict[rel] = {}
            labels = rel_labels[rel]
            if rel_unc:

                for u in ['al','ep']:
                    pred_rel_unc = pred[rel+'_'+u+'_uc'].cpu()
                    unc_vals.rel_batch_unc[rel][u] = pred_rel_unc

                    batch_unc = get_cls_rel_uncertainty(pred_rel_unc,labels,rel)
                    tmp_dict[rel][u] = batch_unc.numpy()

                    for j,l in enumerate(labels):
                        for k in l:
                            unc_vals.cls_rel_uc[rel][k][u].append(pred_rel_unc[j,k].item())
            else:
                batch_unc = torch.zeros(len(rel_labels[rel]),len(list(unc_vals.cls_rel_uc[rel].keys())))
                for i in range(len(rel_labels[rel])):
                    for k in labels[i]:
                        batch_unc[i,k] = 1
                tmp_dict[rel]['al'] = batch_unc.numpy()


        unc_vals.unc_list_rel[index] = tmp_dict
    model.train()
        # if obj_unc or rel_unc:
        #     unc_list_rel,unc_list_obj = normalize_uncertainty(unc_list_rel,cls_rel_uc,
        #                                                 unc_list_obj,cls_obj_uc,
        #                                                 obj_unc,background_mem)


def get_cls_rel_uncertainty(pred_unc,labels,rel_type):
    if rel_type == 'attention':
        labels = torch.tensor(labels, dtype=torch.long).squeeze()
        rel_cls_uc = pred_unc[torch.arange(pred_unc.size(0)),labels]
        batch_unc = torch.zeros(pred_unc.shape)
        batch_unc[torch.arange(batch_unc.size(0)),labels] = rel_cls_uc

    else:
        batch_unc = torch.zeros(pred_unc.shape)
        for i in range(len(labels)):
            for k in labels[i]:
                batch_unc[i,k] = pred_unc[i,k]
    return batch_unc

def normalize_batch_uncertainty(unc_list_rel,cls_rel_uc,
                          unc_list_obj,cls_obj_uc,obj_unc =False, background_mem=False, weight_type=['both']):

    
    for u in weight_type:

        if obj_unc:
            if u == 'both':
                batch_unc = unc_list_obj['al'] + unc_list_obj['ep']
            else:
                batch_unc = unc_list_obj[u]

           
            index,batch_classes = np.where(batch_unc!=0)

            unq_batch_classes = np.unique(batch_classes)
            for k in  unq_batch_classes:
                if not background_mem:
                    cls = k + 1
                else:
                    cls = k
                unq_idx = np.where(batch_classes == k)[0]
                # batch_unc[index[unq_idx],batch_classes[unq_idx]] = (batch_unc[index[unq_idx],batch_classes[unq_idx]]- cls_obj_uc[cls][u][0])/(cls_obj_uc[cls][u][1] + 1e-12)
                batch_unc[index[unq_idx],batch_classes[unq_idx]] = np.exp(batch_unc[index[unq_idx],batch_classes[unq_idx]])/(cls_obj_uc[cls][u] + 1e-12)


            unc_list_obj[u] = batch_unc

        for rel in ['attention','spatial','contacting']:
            
            if u == 'both':
                batch_unc = unc_list_rel[rel]['al'] + unc_list_rel[rel]['ep']
            else:
                batch_unc = unc_list_rel[rel][u]

            index,batch_classes = np.where(batch_unc!=0)
            unq_batch_classes = np.unique(batch_classes)
            for k in  unq_batch_classes:
                unq_idx = np.where(batch_classes == k)[0]
                # batch_unc[index[unq_idx],batch_classes[unq_idx]] = (batch_unc[index[unq_idx],batch_classes[unq_idx]]- cls_rel_uc[rel][k][u][0])/ (cls_rel_uc[rel][k][u][1] + 1e-12)
                batch_unc[index[unq_idx],batch_classes[unq_idx]] = np.exp(batch_unc[index[unq_idx],batch_classes[unq_idx]])/ (cls_rel_uc[rel][k][u] + 1e-12)
            

            unc_list_rel[rel][u] = batch_unc
    return unc_list_rel,unc_list_obj
