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

# Transition is a namedtuple which includes 2 objects, Transition and a sasr tuple
# the 'Transition' defines the type
# example
    # Person = collections.namedtuple('Person', 'name age gender')
    # print 'Type of Person:', type(Person)
    # >> Type of Person: <type 'type'>
    # bob = Person(name='Bob', age=30, gender='male')
    # print bob
    # >> Person(name='Bob', age=30, gender='male')
Transition = namedtuple('Transition',
                            (
                                'img_state',
                                'numerical_state',
                                'action',
                                'next_img_state',
                                'next_numerical_state',
                                'reward'
                            ))

# define the simple problem to be as follows
# state = the location of the end effector and the angles
# action = the radial acceleration of each joint
# reward = time it takes to arrive at the target

# the class responsible for experience replay
# memory = a list of Transition namedtuple
class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        # if the memory is less than the capacity, add one element to the list
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        # else, add a memory to the position
        self.memory[self.position] = Transition(*args)
        # take care for the position wrap around
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        # sample from the memory
        return random.sample(self.memory, int(batch_size))

    def __len__(self):
        # returns the length of the memory
        return len(self.memory)

def output_shape(in_shape,k,s,p):
    return ((in_shape-k+2*p)//s)+1

# sirapoab also helped in this commit
# this model will output the Q-value for each of the action
# the action is defined as -1,0,1 for each joint, so there will be 3*3*3 = 27 available actions
# the states will be the andgle of each joint: 3, and the endpoint: 2, the targets: 2 = 3+2+2=7 states
class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        N_AVAIL_ACTIONS = 3
        N_DIM_STATES = 6
        N_IN_CHANNEL = 3
        N_IN_SHAPE = 64

        self.conv0 = nn.Conv2d(N_IN_CHANNEL, out_channels=16, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.conv1 = nn.Conv2d(16, out_channels=16, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.conv2 = nn.Conv2d(16, out_channels=16, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.conv3 = nn.Conv2d(16, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.conv4 = nn.Conv2d(32, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.conv5 = nn.Conv2d(32, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.conv6 = nn.Conv2d(64, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.conv7 = nn.Conv2d(64, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.conv8 = nn.Conv2d(64, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.conv9 = nn.Conv2d(64, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)

        self.fc_00 = nn.Linear(in_features=256,out_features=256)
        self.fc_01 = nn.Linear(in_features=256,out_features=64)

#        self.pool3 = nn.MaxPool2d(3)
        self.pool2 = nn.MaxPool2d(2)

        self.convsmall0 = nn.Conv2d(64, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.convsmall1 = nn.Conv2d(64, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.convsmall2 = nn.Conv2d(64, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.convsmall3 = nn.Conv2d(64, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.convsmall4 = nn.Conv2d(64, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.convsmall5 = nn.Conv2d(64, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)

        self.fc_end0 = nn.Linear(in_features=64, out_features=64, bias=True)
        self.fc_end1 = nn.Linear(in_features=64, out_features=64, bias=True)
        self.fc_end2 = nn.Linear(in_features=64, out_features=1, bias=True)

        self.activation = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        # ========================================================================
        self.N_AVAIL_ACTIONS = 5 # 3 translation, 3 orientation, 2 close/open
        self.N_DIM_STATES = 9 # 6 for hand, 6 for target, 1 gripper status

        self.fc_10 = nn.Linear(in_features=8,out_features=256)
        self.fc_11 = nn.Linear(in_features=256,out_features=256)
        self.fc_12 = nn.Linear(in_features=256,out_features=64)

    def forward(self,img_state,numerical_state,action):
#        x = torch.cat([state[:,3:],action],dim=1)
        img_state = img_state.float()/255.00
#        print("dqn: img_state",img_state)
        img = self.activation(self.conv0(img_state))
        img = self.activation(self.conv1(img))
        img = self.pool2(img) #/2 32
        img = self.activation(self.conv2(img))
        img = self.activation(self.conv3(img))
        img = self.pool2(img) #/4 16
        img = self.activation(self.conv4(img))
        img = self.activation(self.conv5(img))
        img = self.pool2(img) #/8 8
        img = self.activation(self.conv6(img))
        img = self.activation(self.conv7(img))
        img = self.pool2(img) #/8 4
        img = self.activation(self.conv8(img))
        img = self.activation(self.conv9(img))
        img = self.pool2(img) #/8 2

#        img = img.flatten(1)
#        img = self.activation(self.fc_00(img))
#        img = self.activation(self.fc_01(img))

        # get the gripper open/close, and get the gripper height
#        act = torch.cat([numerical_state[:,[2,-2,-1]],action],dim=1)# z gripper, has target, grasp
        act = torch.cat([numerical_state[:,[2,-1]],action],dim=1)# z gripper, has target, grasp
        act = self.activation(self.fc_10(act))
        act = self.activation(self.fc_11(act))
        act = self.activation(self.fc_12(act))
        act = act.unsqueeze(-1).unsqueeze(-1)

        img = img + act

        img = self.activation(self.convsmall0(img))
        img = self.activation(self.convsmall1(img))
        img = self.pool2(img) #4
        img = self.activation(self.convsmall2(img))
        img = self.activation(self.convsmall3(img))
        img = self.pool2(img) #2
        img = self.activation(self.convsmall4(img))
        img = self.activation(self.convsmall5(img))
        img = self.pool2(img) #1
        img = img.flatten(1) #flatten from the 1st dimension to the last dimension

#        img = torch.cat([img,act],dim=1)
        img = self.activation(self.fc_end0(img))
        img = self.activation(self.fc_end1(img))
        img = self.fc_end2(img)

        return img
