from __future__ import print_function

import vrep_sawyer
import simulator
import dqn
import valuenet
import tqdm
import bbopt
import brain
import twodrobot

import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")

#==================================================
# define the robotics arm
#==================================================

# attach it to a simulator
dt = 100e-3
# r = twodrobot.TwoDRobot(dt)
r = vrep_sawyer.VrepSawyer(dt)
s = simulator.Simulator(r,dt,target_x=0,target_y=0,target_z=0,visualize=False)

t = 0

#=================================================
# define the DQN
#=================================================
MODEL_NAME = "vrep_arm_model.pt"
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
STATE_IS_IMAGE = False
CONTINUE_TRAINING = True
MEMORY_SIZE = 1000000

# create the target and policy networks
policy_net = dqn.DQN().to(device)
# default xavier init
for m in policy_net.modules():
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform(m.weight, gain=nn.init.calculate_gain('relu'))

# if the model file exists, load it
if os.path.isfile("vrep_arm_model.pt") and CONTINUE_TRAINING:
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
    simulator=s, #only to access scripted policy
    policy_net=policy_net,
    target_net1=target_net1,
    target_net2=target_net2,
    memory_size=MEMORY_SIZE,
    value_net_trainer=value_net_trainer,
    state_is_image=STATE_IS_IMAGE,
    use_value_net=USE_VALUE_NET)


def test_model(iterations,criteria):
    print("========= agent test ========")
    s.set_visualize(True)
    scores = np.zeros(iterations)
    reached = 0
    for i in tqdm.tqdm(range(iterations)):
        s.reset()
        target_x,target_y,target_z = s.randomly_place_target()
        _, state = s.get_robot_state()
        for t in count():
            # Select and perform an action based on epsilon greedy
            # action is chosen based on the policy network
            state = torch.Tensor(state)

            # get the reward, detect if the task is done
            action = None
            last_state = state
            action = br.select_action_epsilon_greedy(state,epsilon = 0) #action is a tensor

            s.set_control(action.view(-1).cpu().numpy())
            s.step()
            _, state = s.get_robot_state()
            reward,done = s.get_reward_and_done(state)
            reward = torch.tensor([reward], device=device)

            if done and reward>0:
                #reached the target on its own, and not done because it messed up
                reached += 1
            if t>MAX_TIME:
                # we will terminate if it doesn't finish
                done = True
            if done:
                break
        # lazy evaluation
        if(reached*1.0/iterations > criteria):
            print("lazy eval: agent passes the test :D")
            print("========= end of test ========")
            test_results.append(reached*1.0/iterations)
            plot_test_results()
            return True
        if (((iterations-i)+reached)*1.0/iterations < criteria):
            print("lazy eval: agent fails the test :(")
            print("========= end of test ========")
            test_results.append(reached*1.0/iterations)
            plot_test_results()
            return False
    print("test completed, success rate =",reached*1.0/iterations)
    test_results.append(reached*1.0/iterations)
    plot_test_results()
    if(reached*1.0/iterations > criteria):
        print("agent passes the test :D")
        print("========= end of test ========")
        return True
    else:
        print("agent fails the test :(")
        print("========= end of test ========")
        return False

episode_durations = []
test_results = []

def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('time left')
    plt.xlabel('Episode')
    plt.ylabel('timestep')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated

def plot_test_results():
    plt.figure(1)
    plt.clf()
    test_results_t = torch.tensor(test_results, dtype=torch.float)
    plt.title('Test results')
    plt.xlabel('Episode')
    plt.ylabel('score')
    plt.plot(test_results_t.numpy())
    # Take 100 episode averages and plot them too
    if len(test_results_t) >= 100:
        means = test_results_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated


# ============================================================================
# train for num_episodes epochs
# ============================================================================
from itertools import count

reached = 0
INDEX_OF_ENDPOINT_X = 3
INDEX_OF_ENDPOINT_Y = 4
INDEX_OF_target_X = 5
INDEX_OF_target_Y = 6
MAX_TIME = 200
SEE_EVERY = 100
TARGET_UPDATE = 20
FRAME_SKIP = 1
TEST_EVERY = 100
TEST_QUESTIONS = 10
EXPLORE_EVERY = 5
is_bootstrapping = False
s.set_visualize(True)

num_episodes = 10000
for i_episode in range(num_episodes):
    print("training: episode",i_episode)
    # Initialize the environment and state
    s.reset()
    target_x,target_y,target_z = s.randomly_place_target()
    state = s.get_robot_state()
    # we will train the value net at the beginning of each episode
    error = np.random.normal(0,0.1)
    if USE_VALUE_NET:
        value_net_trainer.enable_training()

    for t in count():
        # Select and perform an action based on epsilon greedy
        # action is chosen based on the policy network
        state = torch.Tensor(state)

        # get the reward, detect if the task is done
        a = [0,0,0]
        action = None
        last_state = state
        if(is_bootstrapping and (i_episode % EXPLORE_EVERY != 0)):
            action = br.select_action_scripted_exploration(thresh=1.0,error=error)
        else:
            action = br.select_action_epsilon_greedy(state)

#        print("action =",action.view(-1).numpy())
        s.set_control(action.view(-1).cpu().numpy())
        s.step()
        state = s.get_robot_state()
        reward_number,done = s.get_reward_and_done(state)
        reward = torch.tensor([reward_number], device=device)

        if done and reward_number > 0:
            #reached the target on its own
            reached += 1

        if t>MAX_TIME:
            # we will terminate if it doesn't finish
            done = True

        # Observe new state
        if not done:
            state_tensor = torch.Tensor(state)
#            print('state tensor', state_tensor.size())
        else:
            state_tensor = None

        # Store the transition in memory
        # as the states are ndarray, change it to tensor
        # the action and rewards are already tensors, so they're cool
        if (t%FRAME_SKIP == 0):
            if (t%2 == 0):
                br.memory.push(torch.Tensor(last_state), action.view(-1).float(), state_tensor, reward)
            else:
                br.memory_online.push(torch.Tensor(last_state), action.view(-1).float(), state_tensor, reward)

        br.optimize_model()

        if done:
            #visualize and break
            episode_durations.append(MAX_TIME-(t + 1))
            plot_durations()
            break

    s.set_visualize(False)
    if i_episode % SEE_EVERY == 0:
        s.set_visualize(True)

    if i_episode % 10 == 0:
        print("reached target",reached,"/ 10 times ")
        reached = 0

    if i_episode % TEST_EVERY == 0 and is_bootstrapping:
        # test to see if we have bootstrapped enough to switch back to qtopt
        # test model for 10 iterations to see if it has passed the test
#        has_passed_test = test_model(iterations=TEST_QUESTIONS,criteria=0.5)
        has_passed_test = True #override it into an online trainer instead
        if has_passed_test:
            print("Test passed! switching to QTOpt")
            is_bootstrapping = False

    # Update the target network, copying all weights and biases in DQN
    # considering the fact that we're updating every 6000 timestep
    if i_episode % (6000/MAX_TIME) == 0:#(TARGET_UPDATE*FRAME_SKIP) == 0:
        br.update_target_net()

    if i_episode % 100 == 0:
        print('> saving model just in case')
        torch.save(br.policy_net.state_dict(), MODEL_NAME)
        print("> model saved!")

print('Complete')
torch.save(br.policy_net.state_dict(), MODEL_NAME)
print("model saved!")
plt.ioff()
plt.show()
