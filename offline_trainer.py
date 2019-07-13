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
CONTINUE_TRAINING = False
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
N_TIMESTEP = N_EPISODES * TIMESTEP_PER_EPISODE
SWITCH_EVERY = TIMESTEP_PER_EPISODE * 50
PRINT_EVERY = TIMESTEP_PER_EPISODE * 1
SAVE_EVERY = TIMESTEP_PER_EPISODE * 100
ONLINE_PLOT_EVERY = TIMESTEP_PER_EPISODE * 200
FILE_PREFIX = "dataset/replay_"
n_remaining_timesteps = N_TIMESTEP
is_finetuning = False
MAX_N_MEMORY_FILES = 50
# launch a thread to collect online data
online_thread = datacollection_thread.DataCollectionThread("DataCollectionThread", br.memory_online, maxtime=20, dt=0.05, port=19991, epsilon=0.2)
online_thread.daemon = True # daemon thread exits with the main thread, which is good
online_thread.start()

def switch_memory():
    N_MEMORY_FILES = len(os.listdir('dataset'))
    num_file = np.random.randint(0,N_MEMORY_FILES)
    filename = FILE_PREFIX + str(num_file).zfill(2) + ".gz"

    with gzip.open(filename, 'rb') as handle:
        print("> switching memory ...")
        memory = cPickle.load(handle)
        print("> memory switched to",num_file)
    br.memory = memory

def save_model():
    print("> saving model ...")
    torch.save(br.policy_net.state_dict(), MODEL_NAME+".pt")
    torch.save(br.target_net1.state_dict(), MODEL_NAME+"_target1.pt")
    torch.save(br.target_net1.state_dict(), MODEL_NAME+"_target2.pt")
    print("> model saving completed")

def plot_graph(x,y,title):
    plt.figure(title)
    plt.plot(np.array(x),np.array(y))
    plt.title(title)
    plt.pause(0.001)
    plt.savefig(title+".png")

def reconfigure_for_finetuning():
    global is_finetuning
    global SAVE_EVERY
    global online_thread

    is_finetuning = True
    # increase the rate in which the memory is changed and switched
    online_thread.epsilon = 0.2
    SAVE_EVERY = TIMESTEP_PER_EPISODE * 1
    print("agent graduated, yay!")

currrent_online_file = 0
losses = 0
elapsed_time = 0
success_rates = []
mean_rewards = []
timesteps = []

time_start = time.time()
for i in range(N_TIMESTEP):
    # to print error every episode
    if(i%PRINT_EVERY == 0):
        elapsed_time = time.time() - time_start
        if not is_finetuning:
            print("> Episode",(i/TIMESTEP_PER_EPISODE),", loss =",losses/TIMESTEP_PER_EPISODE,", time =",elapsed_time,"s")
        else:
            print("> Episode (f)",(i/TIMESTEP_PER_EPISODE),", loss =",losses/TIMESTEP_PER_EPISODE,", time =",elapsed_time,"s")
        losses = 0.0
        time_start = time.time()

    # to switch memory to increase memory diversity
    if (i%SWITCH_EVERY == 0):
        switch_memory()

    # to save training checkpoints
    if ((i+1)%SAVE_EVERY == 0):
        save_model()

    if ((i+1)%ONLINE_PLOT_EVERY == 0):
        success_rate = None
        success_rate, mean_reward = online_thread.get_performance()

        if success_rate is not None:
            timesteps.append(i)
            success_rates.append(success_rate)
            mean_rewards.append(mean_reward)
            plot_graph(timesteps,success_rates,"successrate")
            plot_graph(timesteps,mean_rewards,"mean_reward")

        if success_rate > 0.5 and not is_finetuning:
            print("==================================================")
            print("success rate over 0.5, reconfiguring online memory")
            print("==================================================")
            #switch to online training
            #remove all old memories and use only one file now
            reconfigure_for_finetuning()

    # optimize the model
    loss = br.optimize_model()
    losses += loss

    if i % 6000 == 0:#(TARGET_UPDATE*FRAME_SKIP) == 0:
        br.update_target_net()

print("training complete, saving model")
torch.save(policy_net.state_dict(), MODEL_NAME+".pt")
print("model saving completed")

print("end of offline training, beginning online training")
br.memory = dqn.ReplayMemory(br.MEMORY_SIZE)
br.memory_online= dqn.ReplayMemory(br.MEMORY_SIZE)
