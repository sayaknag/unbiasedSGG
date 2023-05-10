# Unbiased Scene Graph generation in Videos
Pytorch Implementation of the framework TEMPURA proposed in our paper [Unbiased Scene Graph Generation in Videos](https://arxiv.org/abs/2304.00733) accepted by **CVPR2023**.

## Abstract
The task of dynamic scene graph generation (SGG) from videos is complicated and challenging due to the inherent dynamics of a scene, temporal fluctuation of model predictions, and the long-tailed distribution of the visual relationships in addition to the already existing challenges in image-based SGG. Existing methods for dynamic SGG have primarily focused on capturing spatio-temporal context using complex architectures without addressing the challenges mentioned above, especially the long-tailed distribution of relationships. This often leads to the generation of biased scene graphs. To address these challenges, we introduce a new framework called TEMPURA: TEmporal consistency and Memory Prototype guided UnceRtainty Attenuation for unbiased dynamic SGG. TEMPURA employs object-level temporal consistencies via transformer-based sequence modeling, learns to synthesize unbiased relationship representations using memory-guided training, and attenuates the predictive uncertainty of visual relations using a Gaussian Mixture Model (GMM). Extensive experiments demonstrate that our method achieves significant (up to 10% in some cases) performance gain over existing methods highlighting its superiority in generating more unbiased scene graphs.

![GitHub Logo](/data/framework.png)

## Requirements
Please install packages in the ```environment.yml``` file.

## Usage

We borrow some compiled code for bbox operations.
```
cd lib/draw_rectangles
python setup.py build_ext --inplace
cd ..
cd fpn/box_intersections_cpu
python setup.py build_ext --inplace
```
For the object detector part, please follow the compilation from https://github.com/jwyang/faster-rcnn.pytorch
We provide a pretrained FasterRCNN model for Action Genome. Please download [here](https://drive.google.com/file/d/1-u930Pk0JYz3ivS6V_HNTM1D5AxmN5Bs/view?usp=sharing) and put it in 
```
fasterRCNN/models/faster_rcnn_ag.pth
```

## Dataset
We use the dataset [Action Genome](https://www.actiongenome.org/#download) to train/evaluate our method. Please process the downloaded dataset with the [Toolkit](https://github.com/JingweiJ/ActionGenome). The directories of the dataset should look like:
```
|-- ag
    |-- annotations   #gt annotations
    |-- frames        #sampled frames
    |-- videos        #original videos
```
 In the experiments for SGCLS/SGDET, we only keep bounding boxes with short edges larger than 16 pixels. Please download the file [object_bbox_and_relationship_filtersmall.pkl](https://drive.google.com/file/d/19BkAwjCw5ByyGyZjFo174Oc3Ud56fkaT/view?usp=sharing) and put it in the ```dataloader```

## Train
+ For PREDCLS: 
```
python train.py -mode predcls -datasize large -data_path $DATAPATH -rel_mem_compute joint -rel_mem_weight_type simple -mem_fusion late -mem_feat_selection manual  -mem_feat_lambda 0.5  -rel_head gmm -obj_head linear -K 6 -lr 1e-5 -save_path output/ 

```

+ For SGCLS: 
```
python train.py -mode sgcls -datasize large -data_path $DATAPATH -rel_mem_compute joint -rel_mem_weight_type simple -mem_fusion late -mem_feat_selection manual  -mem_feat_lambda 0.3  -rel_head gmm -obj_head linear -obj_con_loss euc_con  -lambda_con 1  -eos_coef 1 -K 4 -tracking -lr 1e-5 -save_path output/ 

```
+ For SGDET: 
```
python train.py -mode sgdet -datasize large -data_path $DATAPATH -rel_mem_compute joint -rel_mem_weight_type simple -mem_fusion late -mem_feat_selection manual  -mem_feat_lambda 0.5  -rel_head gmm -obj_head linear -obj_con_loss euc_con  -lambda_con 1  -eos_coef 1 -K 4 -tracking -lr 1e-5 -save_path output/ 

```

## Evaluation
[Trained Models](https://drive.google.com/drive/folders/1m1xSUbqBELpogHRl_4J3ED7tlyp3ebv8?usp=share_link)

+ For PREDCLS: 
```
python test.py -mode predcls -datasize large -data_path $DATAPATH -model_path $MODELPATH -rel_mem_compute joint -rel_mem_weight_type simple -mem_fusion late -mem_feat_selection manual  -mem_feat_lambda 0.5  -rel_head gmm -obj_head linear -K 6 -lr 1e-5  

```

+ For SGCLS: 
```
python test.py -mode sgcls -datasize large -data_path $DATAPATH -model_path $MODELPATH -rel_mem_compute joint -rel_mem_weight_type simple -mem_fusion late -mem_feat_selection manual  -mem_feat_lambda 0.3  -rel_head gmm -obj_head linear -obj_con_loss euc_con  -lambda_con 1  -eos_coef 1 -K 4 -tracking -lr 1e-5 

```
+ For SGDET: 
```
python test.py -mode sgdet -datasize large -data_path $DATAPATH -model_path $MODELPATH -rel_mem_compute joint -rel_mem_weight_type simple -mem_fusion late -mem_feat_selection manual  -mem_feat_lambda 0.5  -rel_head gmm -obj_head linear -obj_con_loss euc_con  -lambda_con 1  -eos_coef 1 -K 4 -tracking -lr 1e-5 

```

## Acknowledgments 
We would like to acknowledge the authors of the following repositories from where we borrowed some code
+ [Yang's repository](https://github.com/jwyang/faster-rcnn.pytorch)
+ [Zellers' repository](https://github.com/rowanz/neural-motifs) 
+ [Cong's repository](https://github.com/yrcong/STTran.git)
