import numpy as np
np.set_printoptions(precision=4)
import copy
import torch
import datetime
import time
from dataloader.action_genome import AG, cuda_collate_fn

####from lib.pseudo_memory_compute import pseudo_memory_computation
from lib.config import Config
from lib.evaluation_recall import BasicSceneGraphEvaluator
from lib.object_detector import detector
from lib.tempura import TEMPURA
from lib.ds_track import get_sequence

conf = Config()

for i in conf.args:
    print(i,':', conf.args[i])

AG_dataset = AG(mode="test", datasize=conf.datasize, data_path=conf.data_path, filter_nonperson_box_frame=True,
                filter_small_box=False if conf.mode == 'predcls' else True)
dataloader = torch.utils.data.DataLoader(AG_dataset, shuffle=False, num_workers=0, collate_fn=cuda_collate_fn)

gpu_device = torch.device('cuda:0')
object_detector = detector(train=False, object_classes=AG_dataset.object_classes, use_SUPPLY=True, mode=conf.mode).to(device=gpu_device)
object_detector.eval()


model = TEMPURA(mode=conf.mode,
               attention_class_num=len(AG_dataset.attention_relationships),
               spatial_class_num=len(AG_dataset.spatial_relationships),
               contact_class_num=len(AG_dataset.contacting_relationships),
               obj_classes=AG_dataset.object_classes,
               enc_layer_num=conf.enc_layer,
               dec_layer_num=conf.dec_layer,
               obj_mem_compute = conf.obj_mem_compute,
               rel_mem_compute = conf.rel_mem_compute,
               take_obj_mem_feat= conf.take_obj_mem_feat,
               mem_fusion= conf.mem_fusion,
               selection = conf.mem_feat_selection,
               selection_lambda=conf.mem_feat_lambda,
               obj_head = conf.obj_head,
               rel_head = conf.rel_head,
               K = conf.K,
               tracking= conf.tracking).to(device=gpu_device)

model.eval()

if conf.save_path is not None:
    log_val = open(conf.save_path+'log_val.txt', mode = 'a')
    log_val.write('-'*30+'all_mode_eval'+'-'*30+'\n')
else:
    log_val = None

ckpt = torch.load(conf.model_path, map_location=gpu_device)
model.load_state_dict(ckpt['state_dict'], strict=True)
#####if conf.pseudo_mem_compute:
    #####rel_memory = pseudo_memory_computation(AG_dataset,dataloader,model,object_detector,gpu_device,conf)
#### if 'object_memory' or 'rel_memory' in ckpt.keys():
#  #####   model.object_classifier.obj_memory = ckpt['object_memory'].to(gpu_device)
#  ####   model.rel_memory = {k:ckpt['rel_memory'][k].to(gpu_device) for k in ckpt['rel_memory'].keys()}
print('*'*50)
print('CKPT {} is loaded'.format(conf.model_path))
#
evaluator1 = BasicSceneGraphEvaluator(
    mode=conf.mode,
    AG_object_classes=AG_dataset.object_classes,
    AG_all_predicates=AG_dataset.relationship_classes,
    AG_attention_predicates=AG_dataset.attention_relationships,
    AG_spatial_predicates=AG_dataset.spatial_relationships,
    AG_contacting_predicates=AG_dataset.contacting_relationships,
    output_dir = conf.save_path,
    iou_threshold=0.5,
    constraint='with')

evaluator2 = BasicSceneGraphEvaluator(
    mode=conf.mode,
    AG_object_classes=AG_dataset.object_classes,
    AG_all_predicates=AG_dataset.relationship_classes,
    AG_attention_predicates=AG_dataset.attention_relationships,
    AG_spatial_predicates=AG_dataset.spatial_relationships,
    AG_contacting_predicates=AG_dataset.contacting_relationships,
    output_dir = conf.save_path,
    iou_threshold=0.5,
    constraint='semi', semithreshold=0.9)

evaluator3 = BasicSceneGraphEvaluator(
    mode=conf.mode,
    AG_object_classes=AG_dataset.object_classes,
    AG_all_predicates=AG_dataset.relationship_classes,
    AG_attention_predicates=AG_dataset.attention_relationships,
    AG_spatial_predicates=AG_dataset.spatial_relationships,
    AG_contacting_predicates=AG_dataset.contacting_relationships,
    output_dir = conf.save_path,
    iou_threshold=0.5,
    constraint='no')

start_time = time.time()

with torch.no_grad():
    for b, data in enumerate(dataloader): 
        print('index: ',data[4], flush=True)
        im_data = copy.deepcopy(data[0].cuda(0))
        im_info = copy.deepcopy(data[1].cuda(0))
        gt_boxes = copy.deepcopy(data[2].cuda(0))
        num_boxes = copy.deepcopy(data[3].cuda(0))
        gt_annotation = AG_dataset.gt_annotations[data[4]]
        video_id = AG_dataset.valid_video_names[data[4]]
        
        entry = object_detector(im_data, im_info, gt_boxes, num_boxes, gt_annotation, im_all=None)

        if conf.tracking:
            get_sequence(entry, gt_annotation, (im_info[0][:2]/im_info[0,2]).cpu().data,conf.mode)
        
        pred = model(entry,phase='test', unc=False) #pred['rel_features']
        evaluator1.evaluate_scene_graph(gt_annotation, dict(pred))
        evaluator2.evaluate_scene_graph(gt_annotation, dict(pred))
        evaluator3.evaluate_scene_graph(gt_annotation, dict(pred))
        #need to save video_pred_dict = video_id-> {frame_id1 : {'triplet_scores':[],
        #                                                           'triplet_labels':[],
        #                                                           'triplet_boxes':[] }} } as video_id/sgg_dict.pkl
         
total_time = time.time() - start_time
total_time_str = str(datetime.timedelta(seconds=int(total_time)))
print('Inference time {}'.format(total_time_str), flush=True)
# if conf.output_dir is not None:
#     with open(conf.output_dir+"log_"+conf.mode+".txt", "a") as f:
#                 f.truncate(0)
#                 f.close()
constraint_type = 'with constraint'
print('-'*10+constraint_type+'-'*10)
evaluator1.print_stats(log_file=log_val)

constraint_type = 'semi constraint'
print('-'*10+constraint_type+'-'*10)
evaluator2.print_stats(log_file=log_val)

constraint_type = 'no constraint'
print('-'*10+constraint_type+'-'*10)
evaluator3.print_stats(log_file=log_val)
