# Unbiased Scene Graph generation in Videos
Pytorch Implementation of the framework TEMPURA proposed in our paper [Unbiased Scene Graph Generation in Videos](https://arxiv.org/abs/2304.00733) accepted for publication to **CVPR2023**.

## Abstract
The task of dynamic scene graph generation (SGG) from videos is complicated and challenging due to the inherent dynamics of a scene, temporal fluctuation of model predictions, and the long-tailed distribution of the visual relationships in addition to the already existing challenges in image-based SGG. Existing methods for dynamic SGG have primarily focused on capturing spatio-temporal context using complex architectures without addressing the challenges mentioned above, especially the long-tailed distribution of relationships. This often leads to the generation of biased scene graphs. To address these challenges, we introduce a new framework called TEMPURA: TEmporal consistency and Memory Prototype guided UnceRtainty Attenuation for unbiased dynamic SGG. TEMPURA employs object-level temporal consistencies via transformer-based sequence modeling, learns to synthesize unbiased relationship representations using memory-guided training, and attenuates the predictive uncertainty of visual relations using a Gaussian Mixture Model (GMM). Extensive experiments demonstrate that our method achieves significant (up to 10% in some cases) performance gain over existing methods highlighting its superiority in generating more unbiased scene graphs.

![GitHub Logo](/data/framework.png)

## Requirements
Please install packages in the requirements.txt file.

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
|-- action_genome
    |-- annotations   #gt annotations
    |-- frames        #sampled frames
    |-- videos        #original videos
```
 

## Train
+ For PREDCLS: 
```
python train.py -mode predcls -datasize large -data_path data/ag/ -rel_mem_compute joint -rel_mem_weight_type simple -mem_fusion late -mem_feat_selection manual  -mem_feat_lambda 0.5  -rel_head gmm -obj_head linear -K 6 -lr 1e-5 -save_path output/ 

```

+ For SGCLS: 
```
python train.py -mode sgcls -datasize large -data_path data/ag/ -rel_mem_compute joint -rel_mem_weight_type simple -mem_fusion late -mem_feat_selection manual  -mem_feat_lambda 0.3  -rel_head gmm -obj_head linear -obj_con_loss euc_con  -lambda_con 1  -eos_coef 1 -K 4 -tracking -lr 1e-5 -save_path output/ 

```
+ For SGDET: 
```
python train.py -mode sgdet -datasize large -data_path data/ag/ -rel_mem_compute joint -rel_mem_weight_type simple -mem_fusion late -mem_feat_selection manual  -mem_feat_lambda 0.5  -rel_head gmm -obj_head linear -obj_con_loss euc_con  -lambda_con 1  -eos_coef 1 -K 4 -tracking -lr 1e-5 -save_path output/ 

```



## Help 
When you have any question/idea about the code/paper. Please comment in Github or send us Email. We will reply as soon as possible.
