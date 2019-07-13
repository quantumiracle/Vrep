import vrep_sawyer
import simulator
import dqn
import valuenet
import tqdm
import bbopt
import brain
import datacollection_thread

import os
import glob
import cv2
import time
import cPickle
import gzip
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
plt.style.use("ggplot")
#=================================================
# define the DQN
#=================================================
MODEL_NAME = "vrep_arm_model"
# check device
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
from collections import namedtuple
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim

if torch.cuda.is_available():
    print("cuda is available :D")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

USE_VALUE_NET = False
STATE_IS_IMAGE = True
CONTINUE_TRAINING = True
MEMORY_SIZE = 1000000

# create the target and policy networks
policy_net = dqn.DQN().to(device)
target_net1 = dqn.DQN().to(device)
target_net2 = dqn.DQN().to(device)

value_net = valuenet.ValueNet().to(device)
value_net_trainer = valuenet.ValueNetTrainer(value_net)
print("number of parameters: ",sum(p.numel() for p in policy_net.parameters() if p.requires_grad))
target_net1.load_state_dict(policy_net.state_dict())
target_net1.eval()
target_net2.load_state_dict(policy_net.state_dict())
target_net2.eval()

# default xavier init
for m in policy_net.modules():
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform(m.weight, gain=nn.init.calculate_gain('relu'))

# if the model file exists, load it
if os.path.isfile("vrep_arm_model.pt") and CONTINUE_TRAINING:
    policy_net.load_state_dict(torch.load('vrep_arm_model.pt'))
    target_net1.load_state_dict(torch.load('vrep_arm_model_target1.pt'))
    target_net2.load_state_dict(torch.load('vrep_arm_model_target2.pt'))
    print("loaded existing model file")

br = brain.Brain(
    simulator=None,
    policy_net=policy_net,
    target_net1=target_net1,
    target_net2=target_net2,
    memory_size=MEMORY_SIZE,
    value_net_trainer=value_net_trainer,
    state_is_image=STATE_IS_IMAGE,
    use_value_net=USE_VALUE_NET)

N_EPISODES = 20000
TIMESTEP_PER_EPISODE = 20
FILE_PREFIX = "dataset/replay_"
N_WORKERS = 3
EPSILON = 0.0
USE_SCRIPTED_POLICY = True

workers = [None]*N_WORKERS
for i in range(N_WORKERS):
    # launch a thread to collect online data
    workers[i] = datacollection_thread.DataCollectionThread(
                    "DataCollectionThread"+str(i),
                    br.memory,
                    maxtime=TIMESTEP_PER_EPISODE,
                    dt=0.05,
                    port=19990+2*i,
                    epsilon=EPSILON,
                    use_scripted_policy=USE_SCRIPTED_POLICY,
                    vrep_file_name='ik_sawyer'+str(i)+'.ttt')
    workers[i].daemon = True # daemon thread exits with the main thread, which is good
    workers[i].start()

while(len(br.memory) < br.capacity):
    print("memory is now",len(br.memory),"percentage filled: {:2f}".format(len(br.memory)*1.0/br.capacity))
    time.wait(10)

filename = FILE_PREFIX+str(0).zfill(2)+".gz"
print("> Memory full, saving file into",filename)
with gzip.GzipFile(filename, 'wb',compresslevel=GZIP_COMPRESSION_LEVEL) as handle:
    cPickle.dump(br.memory, handle, protocol=cPickle.HIGHEST_PROTOCOL)
print("> saving completed")
