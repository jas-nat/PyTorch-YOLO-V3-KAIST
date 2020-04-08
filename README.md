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
Run `plot.py` to plot precision from `test_data_map.py` 
