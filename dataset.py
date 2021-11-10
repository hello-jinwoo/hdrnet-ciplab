'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import math
import numpy as np
import torch
import random
import torch.nn as nn
import torch.nn.init as init
from torch.utils import data
from torch.utils.data import Dataset, DataLoader, dataloader
from torchvision.utils import save_image
import torchvision.transforms as transforms
# from utils import rgb2uvl
from glob import glob

import cv2
epsilon = 0.000001

pi = 3.141592

transform_test = transforms.Compose([
    transforms.ToTensor(), # test : directly to tensor
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # normalize
])

class MultiDataset(Dataset):
    def __init__(self,root,split,input_type,output_type):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        super(Dataset, self).__init__()
        self.root = root
        self.split = split
        self.input_type = input_type    # rgb / uvl
        self.output_type = output_type  # illumination / uv
        self.places = [f for f in glob(os.path.join(root,split,'*','*.tiff')) if '_mcc' in f]
        self.A_size = len(self.places)  # get the size of dataset A
        self.ls = 128

        random.seed(time.process_time())
        np.random.seed(int(time.process_time()))

    def __len__(self):
        """Return the total number of images in the dataset."""
        return self.A_size

    def __getitem__(self, index):
        dir_input = self.places[index]
        dir_input_gen = os.path.join(dir_input).replace("_mcc.tiff", "")
        dir_ilu = os.path.join(dir_input_gen + "_ilu.tiff")

        num_light_dir = dir_input_gen.split('numlight')[0]
        num_light_str = dir_input_gen.split('_')[-2]
        num_light = int(os.path.join(num_light_str).replace("numlight", ""))

        num_light_type = dir_input_gen.split('_')[::-1][0]
        num_light_num = len(num_light_type)

        input = np.array(cv2.cvtColor(cv2.imread(dir_input), cv2.COLOR_BGR2RGB), dtype=np.float32)
        input_ls = cv2.resize(input, (self.ls, self.ls))
        ilu = np.array(cv2.cvtColor(cv2.imread(dir_ilu), cv2.COLOR_BGR2RGB), dtype=np.float32)
        ilu_norm = ilu / ilu[:,:,1:2]           # normalize green to 1
        ilu_norm_rb = np.delete(ilu_norm, 1, axis=2)

        gt_rgb = input / ilu_norm                   # apply wb

        ret_dict = {}

        input_rgb = transforms.ToTensor()(input)
        input_rgb_lowres = transforms.ToTensor()(input_ls)
        gt_rgb = transforms.ToTensor()(gt_rgb)
        gt_illum = transforms.ToTensor()(ilu_norm_rb)

        return input_rgb_lowres / 255., input_rgb / 255., gt_rgb / 255., gt_illum, dir_input

def get_loader(config, split):
    dataset = MultiDataset(root=config.data_root,
                           split=split,
                           input_type=config.input_type,
                           output_type=config.output_type)
    if split == 'test':
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False,
                                num_workers=config.num_workers)
    else:
        dataloader = DataLoader(dataset, batch_size=config.batch_size,
                                 shuffle=True,num_workers=config.num_workers)
    return dataloader

if __name__ == '__main__':
    dataset = MultiDataset(root='../data/LSMI_refined',
                           split='train',
                           input_type='uvl',
                           output_type='uv')
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)

    for batch in dataloader:
        print(batch)
        print(batch["input_uvl"].shape)
        print(batch["gt_uv"].shape)
        print(batch["gt_illum"].shape)
        print(batch["mask"].shape)
        input()
    
