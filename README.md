# Unbiased Scene Graph generation in Videos
Pytorch Implementation of TEMPURA.


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
