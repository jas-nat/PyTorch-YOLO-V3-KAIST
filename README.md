# Disclaimer
I stop working on this repository. If you would like to see further updates, please take a look [my new repository](https://github.com/jas-nat/yolov3-KAIST)

# PyTorch YOLO V3 KAIST
Hi! This repository was made for doing my final project of my undergraduate program. Try to apply PyTorch YOLO-V3 from [eriklindernoen](https://github.com/eriklindernoren/PyTorch-YOLOv3) with modification for KAIST Dataset. Now it supports 4 channels (RGB + Infrared) images. 

## Installation
```
$ git clone https://github.com/jas-nat/PyTorch-YOLO-V3-KAIST.git
$ cd PyTorch-YOLO-V3-KAIST/
$ sudo pip3 install -r requirements.txt
```

##### Download pretrained weights
if you wan use pretrained darknet-53 on IMAGENET weights, please download [darknet53.conv.74](https://pjreddie.com/media/files/darknet53.conv.74),and put it into `weights/`

## Conversion Label to YOLO format
Go to `label_transform` and find a code to transform  the annotations to YOLO format. Check the flag and directories inside the file depending on your directories. 

## To train
There are 2 training files. The first one is `train-kaist_single.py` which is mainly based on [packyan] (https://github.com/packyan/PyTorch-YOLOv3-kitti) and `train-kaist_all.py` which is based on the orignal author [eriklindernoen](https://github.com/eriklindernoren/PyTorch-YOLOv3). If you want to do training on RGB or infrared images, use `train-kaist_single.py`, and if you want to do 4 channels at the same time (RGB+Infrared / Multispectral), use `train-kaist_all.py`. 

Please be aware of the followings:
1. Configure `.cfg` and `.data` file in `config`
-channels: 4 for multispectral, 3 for RGB, and 1 for infrared only
-location of training files and validations 
2. Run the training files

## Testing Detection with real image samples
You can use `detect.py` to run some samples after finishing the training. 
Please be aware of the followings:
1. Load the correct weights and directory for sample images
2. Write the correct name for the output directories

## Plot Precision
Run `plot.py` to plot precision from `test_data_map.txt` 

## Notice
I am not an expert of this field yet. I am just doing it for my own research purporse. If you ask questions, I will try my best to answer it.

## Paper
### YOLOv3: An Incremental Improvement
_Joseph Redmon, Ali Farhadi_ <br>

**Abstract** <br>
We present some updates to YOLO! We made a bunch
of little design changes to make it better. We also trained
this new network that’s pretty swell. It’s a little bigger than
last time but more accurate. It’s still fast though, don’t
worry. At 320 × 320 YOLOv3 runs in 22 ms at 28.2 mAP,
as accurate as SSD but three times faster. When we look
at the old .5 IOU mAP detection metric YOLOv3 is quite
good. It achieves 57.9 AP50 in 51 ms on a Titan X, compared
to 57.5 AP50 in 198 ms by RetinaNet, similar performance
but 3.8× faster. As always, all the code is online at
https://pjreddie.com/yolo/.

[[Paper]](https://pjreddie.com/media/files/papers/YOLOv3.pdf) [[Original Implementation]](https://github.com/pjreddie/darknet)


## Credit
```
@article{yolov3,
  title={YOLOv3: An Incremental Improvement},
  author={Redmon, Joseph and Farhadi, Ali},
  journal = {arXiv},
  year={2018}
}
```
