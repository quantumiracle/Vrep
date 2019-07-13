import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import cPickle
import gzip

# 20 timesteps per episode, 5000 episodes per dataset
with gzip.open("dataset/replay_01.gz", "rb") as handle:
    demonstration_data = cPickle.load(handle)
batch_size=3
transitions = demonstration_data.sample(batch_size)  
print(transitions)
for i in range(10):
    numerical_state = transitions[i].numerical_state
    print(numerical_state.shape)

            # 'img_state',
            # 'numerical_state',
            # 'action',
            # 'next_img_state',
            # 'next_numerical_state',
            # 'reward'


