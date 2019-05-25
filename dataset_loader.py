import scipy.misc as m
import scipy.io
import numpy as np
import json
import torch
from random import shuffle
from torch.utils import data

class EgoHandsDatasetLoader(data.Dataset):
    def __init__(self, dataset_root="", datafile="", device=None, shuffle=True):
        self.dataset_root = dataset_root
        self.datafile = datafile
        self.shuffle = shuffle
        self.device = device
        self.load_data(datafile)

    def load_data(self, datafile):
        with open(datafile, 'r') as fp:
            self.data = json.load(fp)
        self.jsonkeys = list(self.data)
        if self.shuffle:
            shuffle(self.jsonkeys)

    def __len__(self):
        """__len__"""
        return len(self.data)

    def __getitem__(self, idx):
        """__getitem__
        :param index:
        """
        source_img_path = self.jsonkeys[idx]
        img = m.imread(self.dataset_root + source_img_path)
        img = np.array(img, dtype=np.uint8)
        # img = img / 255.0

        dataval = self.data[source_img_path]
        frame_num = int(dataval.split("-")[0]) # Frame index num, starts at 0
        target_mat_file = dataval.split("-")[1] # Mat file

        mat = scipy.io.loadmat(self.dataset_root + target_mat_file)
        lbl = mat['vid_hand_masks']
        lbl_data = np.zeros((1, 720, 1280)).astype(float)
        lbl_data[0,:,:] = lbl[:,:,frame_num]

        # img =torch.tensor(img, device=self.device).float()
        # lbl = torch.tensor(lbl, device=self.device).float()

        return img, lbl_data
