import math
import random
import numpy as np
from tqdm import tqdm
from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from dqn import DQN

class ValueNet(DQN):
    def __init__(self):
        super(ValueNet, self).__init__()

        # ========================================================================
        N_DIM_STATES = 13

        self.fc_0 = nn.Linear(in_features=N_DIM_STATES,out_features=300)
        self.fc_1 = nn.Linear(in_features=300,out_features=300)
        self.fc_2 = nn.Linear(in_features=300,out_features=200)
        self.fc_3 = nn.Linear(in_features=200,out_features=1)

    def forward(self,state):
        x = self.activation(self.fc_0(state))
        x = self.activation(self.fc_1(x))
        x = self.activation(self.fc_2(x))
        x = self.fc_3(x)
        return x

    def forward_image(self,state):
        batch_size,n_channel,n_row,n_col = state.size()

        img = self.activation(self.conv0(state))
        img = self.activation(self.conv1(img))
        img = self.pool2(img) #64
        img = self.activation(self.conv2(img))
        img = self.activation(self.conv3(img))
        img = self.pool2(img) #32
        img = self.activation(self.conv4(img))
        img = self.activation(self.conv5(img))
        img = self.activation(self.conv6(img))
        img = self.pool2(img) #16

        img = self.activation(self.convsmall0(img))
        img = self.activation(self.convsmall1(img))
        img = self.pool2(img) #8
        img = self.activation(self.convsmall2(img))
        img = self.activation(self.convsmall3(img))
        img = self.pool2(img) #4

        img = img.view(batch_size,-1)
        img = self.activation(self.fc_end0(img))
        img = self.fc_end1(img)

        return img

class ValueNetTrainer:
    def __init__(self,model):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(),lr=1e-3)
        self.data_train = None
        self.data_test = None
        self.labels_train = None
        self.labels_test = None
        self.lossfn = F.smooth_l1_loss
        self.should_train_net = True

    def enable_training(self):
        self.should_train_net = True

    def disable_training(self):
        self.should_train_net = False

    def set_data(self,data):
        # the data is already shuffled, so we're good
        n_batch = data.size()[0]
        cutoff = int(0.7*n_batch)
        self.data_train = data[:cutoff]
        self.data_test = data[cutoff:]

    def set_labels(self,labels):
        n_batch = labels.size()[0]
        cutoff = int(0.7*n_batch)
        self.labels_train = labels[:cutoff]
        self.labels_test = labels[cutoff:]

    def train(self,epochs = 50):
        #train on only one batch of data (that is sampled to train with the model)
        for i in tqdm(range(epochs)):
            y_pred = self.model(self.data_train)
            loss = self.lossfn(self.labels_train,y_pred)

            self.optimizer.zero_grad()
            loss.backward()

            for param in self.model.parameters():
                # somehow some weights in some layers does not have gradients, i think it is the pooling layers
                if(param.grad is not None):
                    param.grad.data.clamp_(-1, 1)
            self.optimizer.step()

    def evaluate(self):
        with torch.no_grad():
            y_pred = self.model(self.data_test)
            loss = self.lossfn(self.labels_test,y_pred)
            return loss.sum().item()/self.labels_test.sum().item()*100
