# PyTorch YOLO V3 KAIST
Try to apply PyTorch YOLO-V3 from [eriklindernoen](https://github.com/eriklindernoren/PyTorch-YOLOv3) with modification for KAIST Dataset. Now it supports 4 channels (RGB + Infrared) images. 

## Installation
```
$ git clone https://github.com/jas-nat/PyTorch-YOLO-V3-KAIST.git
$ cd PyTorch-YOLO-V3-KAIST/
$ sudo pip3 install -r requirements.txt
```

## Conversion Label to YOLO format
Go to `label_transform` and find a code to transform  the annotations to YOLO format. Check the flag and directories inside the file depending on your directories. 

## To train
1. Configure `.cfg` and `.data` file in `config`
2. Run `train-kaist_all.py` 

## Plot Precision
Run `plot.py` to plot precision from `test_data_map.txt` 

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
