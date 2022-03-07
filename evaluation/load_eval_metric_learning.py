import os
import sys
import pickle
import argparse
import torch
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import numpy as np
import utils
import models
import torchvision
import scipy
import pandas as pd
import torch.nn.functional as F

from scipy import io
from torch import nn
from PIL import Image, ImageFile
from torchvision import models as torchvision_models
from torchvision import transforms as pth_transforms
from eval_knn import extract_features

def calc_recall_at_k(T, Y, k):
    """
    T : [nb_samples] (target labels)
    Y : [nb_samples x k] (k predicted labels/neighbours)
    """

    s = 0
    for t,y in zip(T,Y):
        if t in torch.Tensor(y).long()[:k]:
            s += 1
    return s / (1. * len(T))

dataset = 'sop'

query_features, query_labels = np.load('features/ibatt_sop_query_features.npy'), np.load('features/ibatt_sop_query_labels.npy')

query_features = torch.from_numpy(query_features)
query_labels = torch.from_numpy(query_labels)

if dataset == 'sop':
    K = 1000
    Y = []
    xs = []
    for x in query_features:
        if len(xs)<10000:
            xs.append(x)
        else:
            xs.append(x)
            xs = torch.stack(xs, dim=0)
            cos_sim = torch.mm(xs, query_features.T)
            #cos_sim = F.linear(xs, query_features.T)
            y = query_labels[cos_sim.topk(1 + K)[1][:,1:]]
            Y.append(y.float().cpu())
            xs = []
    
    xs = torch.stack(xs, dim=0)
    cos_sim = torch.mm(xs, query_features.T)
    y = query_labels[cos_sim.topk(1 + K)[1][:,1:]]
    Y.append(y.float().cpu())
    Y = torch.cat(Y, dim=0)

    recall = []
    for k in [1, 10, 100, 1000]:
        r_at_k = calc_recall_at_k(query_labels, Y, k)
        recall.append(r_at_k)
        print("R@{} : {:.3f}".format(k, 100 * r_at_k))