# check device
import vrep_sawyer
import simulator
import dqn
import bbopt

import time
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")

import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count, product

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

ACTION_BOUNDS = np.array([
    [-1.0,1.0], # vx
    [-1.0,1.0], # vy
    [-1.0,1.0], # vz
#    [-1.0,1.0], # wx
#    [-1.0,1.0], # wy
#    [-1.0,1.0], # wz
    [-1.0,1.0], # gripper position, close
    [-1.0,1.0]  # gripper position, open
])
CEM_ITER = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("cuda is available :D")
print(device)

def show_all_q(state_to_see):
    N_AVAIL_ACTIONS = ACTION_BOUNDS.shape[0]
    # build a tensor of all actions
    actions_tensor = []
    for i in product([-1,1],repeat=N_AVAIL_ACTIONS):
        actions_tensor.append(list(i))
    actions_tensor = torch.Tensor(actions_tensor).to(device)

    states_tensor = torch.Tensor(state_to_see).view(1,-1).repeat(actions_tensor.size()[0],1).to(device)
#    print("states",states.size(),"actions_tensor",actions_tensor.size())
    q_vals = policy_net(states_tensor,actions_tensor)

    for i,_ in enumerate(product([-1,1],repeat=N_AVAIL_ACTIONS)):
        print('state',torch.Tensor(state_to_see))
        print('action:',actions_tensor[i,:])
        print("q:",q_vals[i])
        print('---------------------------------')
    print("=========================================")

def maximize_q_network(state):
    with torch.no_grad():
        # repeat the states across all dimesions of available acitons
        state_tensor = state.view(1,-1).to(device)

        # get the list of the best action to do
        bbMaximizer = bbopt.BboptMaximizer(policy_net,None,ACTION_BOUNDS,CEM_ITER,state_tensor,state_is_image=False)
        best_action_tensor = bbMaximizer.get_argmax().view(-1).to(device)

        # return as tensor of [[best_action_number]]
        return best_action_tensor

def select_action_epsilon_greedy(state,epsilon=None):
    best_action_tensor = maximize_q_network(state)
    return best_action_tensor

#==================================================
# define the robotics arm
#==================================================
dt = 50e-3
r = vrep_sawyer.VrepSawyer(dt)
s = simulator.Simulator(r,dt,target_x=0,target_y=0,target_z=0,visualize=False)

t=0

# create the target and policy networks
policy_net = dqn.DQN().to(device)
policy_net.load_state_dict(torch.load('vrep_arm_model.pt'))
policy_net.eval()

while True:
    t += dt
    state = s.get_robot_state()
#    print("state",state)
    reward ,done = s.get_reward_and_done(state)
    show_all_q(state)

    if done and reward > 0:
        t = 0
        print("target reached!")
        s.reset()
        _ = s.randomly_place_target()

    if t > 2:
        print("failed")
        t = 0
        s.reset()
        _ = s.randomly_place_target()

    state = torch.Tensor(state).to(device)
    action = select_action_epsilon_greedy(state)
    s.set_control(action.view(-1).cpu().numpy())
    s.step()
#    time.sleep(dt)
