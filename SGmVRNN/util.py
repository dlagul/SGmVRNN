import torch
import os
import torchvision
import torch.utils.data
import torch.utils.data as data
import numpy as np
from model import *
from tqdm import *
import matplotlib.pyplot as plt
import matplotlib as mpl
import glob

class KpiReader(data.Dataset):
    def __init__(self, path):
        super(KpiReader, self).__init__()
        self.path = path
        self.length = len(glob.glob(self.path+'/*.seq'))
        data = []
        for i in range(self.length):
            item = torch.load(self.path+'/%d.seq' % (i+1))
            data.append(item)
        self.data = data


    def __getitem__(self, index):
        kpi_ts, kpi_label, kpi_value = self.data[index]['ts'],self.data[index]['label'],self.data[index]['value']
        return kpi_ts, kpi_label, kpi_value

    def __len__(self):
        return self.length

