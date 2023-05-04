import torch
import numpy as np
from lib.Uncertainty import normalize_batch_uncertainty

def memory_computation(unc_vals,output_dir,rel_class_num,
                       obj_class_num,obj_feature_dim=1024,
                       rel_feature_dim=1936,obj_weight_type='both',
                       rel_weight_type='both',
                       obj_mem=False,obj_unc=False,
                       include_bg_mem = False):
    
    unc_vals.stats2()
    unc_list_rel = unc_vals.unc_list_rel
    unc_list_obj = unc_vals.unc_list_obj
    cls_rel_uc = unc_vals.cls_rel_uc
    cls_obj_uc = unc_vals.cls_obj_uc

    obj_emb_path = output_dir+'obj_embeddings/'
    rel_emb_path = output_dir+'rel_embeddings/'
    if not include_bg_mem:
        obj_class_num = obj_class_num -1
    obj_norm_factor = torch.zeros(obj_class_num)
    obj_memory = torch.zeros(obj_class_num,obj_feature_dim)
    
    rel_norm_factor = {}
    rel_memory = {}
    for rel in rel_class_num.keys():
        rel_norm_factor[rel] = torch.zeros(rel_class_num[rel])
        rel_memory[rel] = torch.zeros(rel_class_num[rel],rel_feature_dim)
    
    
    if obj_weight_type == 'both':
        # obj_all_u = ['al','ep']
        obj_all_u = ['both']
    elif obj_weight_type == 'al':
        obj_all_u = ['al']
    elif obj_weight_type == 'ep':
        obj_all_u = ['ep']
    else:
        obj_all_u = None

    if rel_weight_type == 'both':
        # rel_all_u = ['al','ep']
        rel_all_u = ['both']
    elif rel_weight_type == 'al':
        rel_all_u = ['al']
    elif rel_weight_type == 'ep':
        rel_all_u = ['ep']
    else:
        rel_all_u = None

    
    for i in unc_list_rel.keys() :
        
        rel_features = torch.tensor(np.load(rel_emb_path+str(i)+'.npy',allow_pickle=True))
        
        if not obj_all_u and obj_mem:
            obj_features = torch.tensor(np.load(obj_emb_path+str(i)+'.npy',allow_pickle=True))
            batch_unc = torch.tensor(unc_list_obj[i]['al']) #  obj_num x c 
            
            index,batch_classes = torch.where(batch_unc!=0)
            batch_unc[index,batch_classes] = 1
            obj_memory += torch.matmul(batch_unc.T,obj_features) 

            unq_batch_classes = torch.unique(batch_classes)
            for k in unq_batch_classes:
                unq_idx = torch.where(batch_classes == k)[0]
                obj_norm_factor[k] = obj_norm_factor[k] + torch.sum(batch_unc[index[unq_idx],batch_classes[unq_idx]])
            # for idx,k in zip(index,batch_classes) :
            #     obj_norm_factor[k] = obj_norm_factor[k] + batch_unc[idx,k]

        if not rel_all_u:   
            for rel in rel_class_num.keys():
                batch_unc = torch.tensor(unc_list_rel[i][rel]['al'])
                index,batch_classes = torch.where(batch_unc!=0)
                batch_unc[index,batch_classes] = 1
                rel_memory[rel] += torch.matmul(batch_unc.T,rel_features)
                
                unq_batch_classes = torch.unique(batch_classes)
                for k in unq_batch_classes:
                    unq_idx = torch.where(batch_classes == k)[0]
                    rel_norm_factor[rel][k] = rel_norm_factor[rel][k] + torch.sum(batch_unc[index[unq_idx],batch_classes[unq_idx]])
                # for idx,k in zip(index,batch_classes) :
                #     rel_norm_factor[rel][k] = rel_norm_factor[rel][k] + batch_unc[idx,k]
            
        else:   
            unc_list_rel[i],unc_list_obj[i] = normalize_batch_uncertainty(unc_list_rel[i],cls_rel_uc,
                          unc_list_obj[i],cls_obj_uc,obj_unc=obj_unc, background_mem=include_bg_mem, 
                          weight_type = rel_all_u)   
            
            if obj_unc and obj_mem:
                for u in obj_all_u :
                    batch_unc = torch.tensor(unc_list_obj[i][u]) #  obj_num x c 
                    index,batch_classes = torch.where(batch_unc!=0)
                    obj_memory += torch.matmul(batch_unc.T,obj_features)
                    
                    # unq_batch_classes = torch.unique(batch_classes)
                    
                    # for k in unq_batch_classes:
                    #     unq_idx = torch.where(batch_classes == k)[0]
                    #     obj_norm_factor[k] = obj_norm_factor[k] + torch.sum(batch_unc[index[unq_idx],batch_classes[unq_idx]])

                    # for idx,k in zip(index,batch_classes) :
                    #     obj_norm_factor[k] = obj_norm_factor[k] + batch_unc[idx,k]
            for u in rel_all_u:
                for rel in rel_class_num.keys():
                    batch_unc = torch.tensor(unc_list_rel[i][rel][u])
                    index,batch_classes = torch.where(batch_unc!=0)
                    rel_memory[rel] += torch.matmul(batch_unc.T,rel_features)
                    
                    # unq_batch_classes = torch.unique(batch_classes)
                    # for k in unq_batch_classes:
                    #     unq_idx = torch.where(batch_classes == k)[0]
                    #     rel_norm_factor[rel][k] = rel_norm_factor[rel][k] + torch.sum(batch_unc[index[unq_idx],batch_classes[unq_idx]])

                    # for idx,k in zip(index,batch_classes) :
                    #     rel_norm_factor[rel][k] = rel_norm_factor[rel][k] + batch_unc[idx,k]
          
    if obj_mem and obj_weight_type=='simple':
        tmp = obj_memory
        nz_idx = torch.where(obj_norm_factor!=0) 
        tmp[nz_idx] = tmp[nz_idx]/(obj_norm_factor[nz_idx].unsqueeze(-1).repeat(1,obj_feature_dim))
        obj_memory = tmp
        # obj_memory = obj_memory/(obj_norm_factor.unsqueeze(-1).repeat(1,obj_feature_dim))
    
    if rel_weight_type == 'simple':
        for rel in rel_memory.keys():
            tmp = rel_memory[rel]
            nz_idx = torch.where(rel_norm_factor[rel]!=0) 
            tmp[nz_idx] = tmp[nz_idx]/(rel_norm_factor[rel][nz_idx].unsqueeze(-1).repeat(1,rel_feature_dim))
            rel_memory[rel] = tmp
            
    
    return rel_memory,obj_memory

