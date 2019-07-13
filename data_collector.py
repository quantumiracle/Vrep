from __future__ import print_function

import vrep_sawyer
import dqn
import valuenet
import tqdm
import bbopt
import brain
import simulator
import twodrobot

import time
import optparse
import cv2
import sys
import time
import cPickle
import gzip
import numpy as np

#=================================================
# define the DQN
#=================================================
MODEL_NAME = "vrep_arm_model.pt"
# check device
import math
import random
import numpy as np
import os
from collections import namedtuple
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim

class DataCollector:
    def __init__(self,dt=50e-3,port=19991):
        #self.r = twodrobot.TwoDRobot(dt)
        self.r = vrep_sawyer.VrepSawyer(dt,headless_mode=True,port_num=port)
        self.s = simulator.Simulator(self.r,dt,target_x=0,target_y=0,target_z=0,visualize=False)

    def collect_data(self,use_scripted_policy=True,visualize=False,n_files=None,
                    start_at=None, epsilon=0.0, dt=50e-3, maxtime=20,
                    dryrun=False, memory_capacity=5000):

        t = 0

        if torch.cuda.is_available():
            print("cuda is available :D")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        USE_VALUE_NET = False
        STATE_IS_IMAGE = True
        MEMORY_SIZE = memory_capacity
        GZIP_COMPRESSION_LEVEL = 3
        self.s.set_visualize(visualize)

        policy_net = None
        target_net1 = None
        target_net2 = None
        value_net = None
        value_net_trainer = None

        # create the target and policy networks
        policy_net = dqn.DQN().to(device)
        # default xavier init
        for m in policy_net.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform(m.weight, gain=nn.init.calculate_gain('relu'))

        if os.path.isfile("vrep_arm_model.pt"):
            policy_net.load_state_dict(torch.load('vrep_arm_model.pt'))
            print("loaded existing model file")

        target_net1 = dqn.DQN().to(device)
        target_net2 = dqn.DQN().to(device)
        value_net = valuenet.ValueNet().to(device)
        value_net_trainer = valuenet.ValueNetTrainer(value_net)
        print("number of parameters: ",sum(p.numel() for p in policy_net.parameters() if p.requires_grad))
        target_net1.load_state_dict(policy_net.state_dict())
        target_net1.eval()
        target_net2.load_state_dict(policy_net.state_dict())
        target_net2.eval()

        br = brain.Brain(
            simulator=self.s, #only to access scripted policy
            policy_net=policy_net,
            target_net1=target_net1,
            target_net2=target_net2,
            memory_size=MEMORY_SIZE,
            value_net_trainer=value_net_trainer,
            state_is_image=STATE_IS_IMAGE,
            use_value_net=USE_VALUE_NET)

        # ============================================================================
        # train for num_episodes epochs
        # ============================================================================
        from itertools import count

        total_reached = 0
        reached = 0
        MAX_TIME = maxtime
        FRAME_SKIP = 1
        MAX_FILES = 6
        FILE_PREFIX = "dataset/replay_"
        num_file = 0
        PRINT_EVERY = 1

        br.memory = dqn.ReplayMemory(capacity=MEMORY_SIZE)

        if not use_scripted_policy:
            FILE_PREFIX = "dataset_online/replay_"
            MAX_FILES = int(MAX_FILES/2) # used to be divided by 2, but nvm for now
        num_episodes = 2000000

        if n_files is not None:
            MAX_FILES = n_files

        if start_at is not None:
            num_file += start_at
            MAX_FILES += start_at


        start_time = time.time()
        total_episode_reward = 0

        for i_episode in range(num_episodes):
            episode_reward = 0
            if i_episode % PRINT_EVERY == 0:
                print("recording: episode",i_episode)
            # Initialize the environment and state
            self.s.reset()
            target_x,target_y,target_z = self.s.randomly_place_target()
            img_state, numerical_state = self.s.get_robot_state()
            error = np.random.normal(0,0.1)
            error = 0

            for t in count():
                # Select and perform an action based on epsilon greedy
                # action is chosen based on the policy network
                img_state = torch.Tensor(img_state)
                numerical_state = torch.Tensor(numerical_state)

                # get the reward, detect if the task is done
                a = [0,0,0]
                action = None
                last_img_state = img_state
                last_numerical_state = numerical_state
                # record the action from the scripted exploration
                thresh = None
                action=None
                if use_scripted_policy:
                    action = br.select_action_scripted_exploration(thresh=1.0,error=error)
                else:
                    action = br.select_action_epsilon_greedy(img_state,numerical_state,epsilon)

                self.s.set_control(action.view(-1).cpu().numpy())
                self.s.step()
                img_state, numerical_state = self.s.get_robot_state()
                reward_number,done = self.s.get_reward_and_done(numerical_state)
                reward = torch.tensor([reward_number], device=device)

                episode_reward += (br.GAMMA**t)*reward_number

                if done and reward_number > 0:
                    #reached the target on its own
    #                print("data collector: episode reached at timestep",t)
                    reached += 1

                if t>MAX_TIME:
                    # we will terminate if it doesn't finish
    #                print("data collector: episode timeout")
                    done = True

                # Observe new state
                if not done:
                    state_img_tensor = torch.Tensor(img_state)
                    state_numerical_tensor = torch.Tensor(numerical_state)
                else:
                    state_img_tensor = None
                    state_numerical_tensor = None

                # Store the transition in memory
                # as the states are ndarray, change it to tensor
                # the actoin and rewards are already tensors, so they're cool
                if (t%FRAME_SKIP == 0):
                    br.memory.push(
                        torch.Tensor(last_img_state),
                        torch.Tensor(last_numerical_state),
                        action.view(-1).float(),
                        state_img_tensor,
                        state_numerical_tensor,
                        reward)

                if done:
                    #visualize and break
                    break

            total_episode_reward += episode_reward

            if i_episode % 10 == 0:
                time_per_ep = (time.time() - start_time) / 10.0
                start_time = time.time()
                print("reached target",reached,"/ 10 times, memory:",len(br.memory),"/",MEMORY_SIZE,",",(100.0*len(br.memory)/MEMORY_SIZE),"% full,",time_per_ep,"sec/ep")
                total_reached += reached
                reached = 0

            if len(br.memory) >= br.memory.capacity:
                # if the buffer is full, save it and reset it
                filename = FILE_PREFIX+str(num_file).zfill(2)+".gz"
                if not dryrun:
                    print("> saving file into",filename)
                    with gzip.GzipFile(filename, 'wb',compresslevel=GZIP_COMPRESSION_LEVEL) as handle:
                        cPickle.dump(br.memory, handle, protocol=cPickle.HIGHEST_PROTOCOL)
                    print("> saving completed")
                else:
                    print("> data collector: dryrun, not saving memory")
                num_file += 1
                if(num_file >= MAX_FILES):
                    print("> data_collector: all files collected, closing")
                    print("> total success rate:",(total_reached*1.0/i_episode))
                    print("> mean episode reward:",(total_episode_reward*1.0/i_episode))
                    return total_reached*1.0/i_episode, total_episode_reward*1.0/i_episode

                br.memory = dqn.ReplayMemory(capacity=MEMORY_SIZE)
                print("> data_collector: memory is full, saved as: "+filename)

if __name__ == "__main__":
    parser = optparse.OptionParser()

    parser.add_option("-s", "--scripted", default="True", help="use scripted policy")
    parser.add_option("-v", "--visualize", default="False", help='visualize the data')
    parser.add_option("-n", "--nfiles", default=1, type="int", help='number of files to gather')
    parser.add_option("-a", "--startat", default=0, type="int",help='index to start from')
    parser.add_option("-e", "--epsilon", default=0.0, type="float",help='epsilon in case of epsilon greedy')
    parser.add_option("-m", "--maxtime", default=20, type="int",help='maximum number of steps')
    parser.add_option("-t", "--timestep", default=50e-3, type="float",help='timestep')
    parser.add_option("-d", "--dryrun", default="False", help='do not save the data')
    parser.add_option("-p", "--port", default=19991, type="int", help='port number')
    parser.add_option("-c", "--capacity", default=5000, type="int", help='memory capacity per file')

    opts, args = parser.parse_args()
    print ('arg:', opts)

    dc = DataCollector(dt=opts.timestep, port=opts.port)

    dc.collect_data(
        use_scripted_policy = opts.scripted == "True",
        visualize=opts.visualize == "True",
        n_files=opts.nfiles,
        start_at=opts.startat,
        epsilon=opts.epsilon,
        dt=opts.timestep,
        maxtime=opts.maxtime,
        dryrun=opts.dryrun == "True",
        memory_capacity = opts.capacity
    )
