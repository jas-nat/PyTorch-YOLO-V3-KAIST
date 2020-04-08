import glob
import random
import os
import numpy as np
from PIL import Image

import torch
from utils.augmentations import horisontal_flip
from torch.utils.data import Dataset


import torch.nn.functional as F
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from skimage.transform import resize

import sys
import warnings

def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad

def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image

def random_resize(images, min_size=288, max_size=448):
    new_size = random.sample(list(range(min_size, max_size + 1, 32)), 1)[0]
    images = F.interpolate(images, size=new_size, mode="nearest")
    return images

class ImageFolder(Dataset):
    def __init__(self, folder_path, img_size=416):
        self.files = sorted(glob.glob('%s/*.*' % folder_path))
        self.img_shape = img_size

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        # Extract image as PyTorch Tensor
        img = transforms.ToTensor()(Image.open(img_path))
        #Pad to square resolution
        img, _ = pad_to_square(img, 0)
        #Resize
        img = resize(img, self.img_size)

        return img_path, input_img

    def __len__(self):
        return len(self.files)


class ListDataset(Dataset):
    def __init__(self, list_path_rgb,list_path_ir, augment=True, multiscale=True, img_size=416, normalized_labels=True):
        with open(list_path_rgb, 'r') as file: #reading rgb images
            self.img_files_rgb = file.readlines()
        with open(list_path_ir, 'r') as file: #reading ir images
            self.img_files_ir = file.readlines()
        self.label_files = [path.replace('images', 'labels').replace('.png', '.txt').replace('.jpg', '.txt') for path in self.img_files_rgb] #change list of images with labels
        #self.img_shape = (img_size, img_size)
        self.img_size = img_size
        self.max_objects = 100
        self.augment = augment
        self.multiscale = multiscale
        self.normalized_labels = normalized_labels
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0

    def __getitem__(self, index):

        #---------
        #  Image
        #---------
        #res = transforms.Resize(self.img_shape) #resize as big as 416 as written in the arguments

        img_path_ir = self.img_files_ir[index % len(self.img_files_ir)].rstrip()
        img_ir = transforms.ToTensor()(Image.open(img_path_ir)) #resize the image into 416, then converting it to tensor
        #print(img_ir.shape)

        img_path_rgb = self.img_files_rgb[index % len(self.img_files_rgb)].rstrip()
        img_rgb = transforms.ToTensor()(Image.open(img_path_rgb)) #resize the image into 416, then converting it to tensor
        #print(img_rgb.shape)
        
        #combine the images into 4 channels
        #dim=1 add 1 dimension
        img = torch.cat((img_ir, img_rgb), dim=0) 
        #print(img.shape)

       
        _, h, w, = img.shape
        #print(_,h,w)
        h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)
        #pad to square resolution
        img, pad = pad_to_square(img, 0)
        _, padded_h, padded_w = img.shape

        #datasets original
        # dim_diff = np.abs(h - w)
        # # Upper (left) and lower (right) padding
        # pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        # # Determine padding
        # pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
        # # Add padding
        # input_img = np.pad(img, pad, 'constant', constant_values=128) / 255.
        # padded_h, padded_w, _ = input_img.shape
        # # Resize and normalize
        # input_img = resize(input_img, (*self.img_shape, 4), mode='reflect') #resize back to 4 maybe (?)
        # # Channels-first
        # input_img = np.transpose(input_img, (2, 0, 1))
        # # As pytorch tensor
        # input_img = torch.from_numpy(input_img).float()

        #---------
        #  Label
        #---------

        label_path = self.label_files[index % len(self.img_files_rgb)].rstrip()
        targets = None

        if os.path.exists(label_path):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
            boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))
            # Extract coordinates for unpadded + unscaled image
            x1 = w_factor * (boxes[:, 1] - boxes[:, 3]/2)
            y1 = h_factor * (boxes[:, 2] - boxes[:, 4]/2)
            x2 = w_factor * (boxes[:, 1] + boxes[:, 3]/2)
            y2 = h_factor * (boxes[:, 2] + boxes[:, 4]/2)
            # Adjust for added padding
            x1 += pad[0]
            y1 += pad[2]
            x2 += pad[1]
            y2 += pad[3]
            # Calculate ratios from coordinates
            boxes[:, 1] = ((x1 + x2) / 2) / padded_w
            boxes[:, 2] = ((y1 + y2) / 2) / padded_h
            boxes[:, 3] *= w_factor / padded_w
            boxes[:, 4] *= h_factor / padded_h
        # Fill matrix
       #    filled_labels = np.zeros((self.max_objects, 5))
            targets = torch.zeros((len(boxes), 6))
            targets[:, 1:] = boxes



        #augmentations
        if self.augment:
            if np.random.random() < 0.5:
                img, targets = horisontal_flip(img, targets)

        return img, targets#, img_path

    def collate_fn(self, batch):
        imgs, targets = list(zip(*batch))
        #remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]
        # add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i 
        targets = torch.cat(targets, 0)
        #selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size +1, 32))
        #reisze images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        self.batch_count += 1
        return imgs, targets

    def __len__(self):
        return len(self.img_files_rgb) #only 1 of them
