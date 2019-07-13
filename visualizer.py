import brain

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np

import vrep_sawyer
import simulator
import twodrobot
import dqn
import valuenet
import brain

import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
plt.style.use("ggplot")

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
import optparse

import torch
import torch.nn as nn
import torch.optim as optim


parser = optparse.OptionParser()
parser.add_option("-t", "--hastarget", default="False", help="view if agent has the target")
parser.add_option("-x", "--xtarget", default=0.8, type="float", help='x coordinate of the grasp target')
parser.add_option("-z", "--ztarget", default=0.01, type="float", help='z coordinate of the grasp target')
opts, args = parser.parse_args()
print ('arg:', opts)

if torch.cuda.is_available():
    print("cuda is available :D")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

USE_VALUE_NET = False
STATE_IS_IMAGE = True
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

r = vrep_sawyer.VrepSawyer(0.05,headless_mode=True,port_num=19992)

br = brain.Brain(
    simulator=None, #only to access scripted policy
    policy_net=policy_net,
    target_net1=target_net1,
    target_net2=target_net2,
    memory_size=MEMORY_SIZE,
    value_net_trainer=value_net_trainer,
    state_is_image=STATE_IS_IMAGE,
    use_value_net=USE_VALUE_NET)

#==============================================================================

fig = plt.figure()
ax = fig.gca(projection='3d')

TRAY_W = r.TRAY_DY
TRAY_H = r.TRAY_DX
TRAY_X = r.TRAY_X
TRAY_Y = r.TRAY_Y
DONE_DISTANCE = 0.25

HAS_OBJECT = opts.hastarget == "True"
HAND_IS_CLOSED = 0
HAS_THE_TARGET = 0

X_BEGIN = TRAY_X - TRAY_H/2.0
Y_BEGIN = TRAY_Y - TRAY_W/2.0
Z_BEGIN = 0
X_END = TRAY_X + TRAY_H/2.0
Y_END = TRAY_Y + TRAY_W/2.0
Z_END = DONE_DISTANCE*1.2
STEP_X = 0.04
STEP_Y = 0.08
STEP_Z = 0.04

OBJECT_X = opts.xtarget
OBJECT_Y = 0
OBJECT_Z = opts.ztarget

# x,y,z is a 3d array of x,y, and z value in each of the 3d mesh grid point
x_arm, y_arm, z_arm = np.meshgrid(np.arange(X_BEGIN, X_END, STEP_X),
                      np.arange(Y_BEGIN, Y_END, STEP_Y),
                      np.arange(Z_BEGIN, Z_END, STEP_Z))

if not HAS_OBJECT:
    print("visualizer: doesn't have object")
    x_dist = OBJECT_X - x_arm
    y_dist = OBJECT_Y - y_arm
    z_dist = OBJECT_Z - z_arm
    HAS_THE_TARGET = 0
else:
    print("visualizer: have the object")
    x_dist = np.ones_like(x_arm)*0
    y_dist = np.ones_like(x_arm)*0
    z_dist = np.ones_like(x_arm)*-0.02
    HAS_THE_TARGET = 1
    HAND_IS_CLOSED = 1

BATCH_SIZE = 128

# value is a 3d array with values in each state
print("meshgrid shape:",x_arm.shape)
n_x, n_y, n_z = x_arm.shape
values = np.zeros((n_x,n_y,n_z))
action_x = np.zeros((n_x,n_y,n_z))
action_y = np.zeros((n_x,n_y,n_z))
action_z = np.zeros((n_x,n_y,n_z))
states = []
count = 0
for i in tqdm(range(n_x)):
    for j in tqdm(range(n_y)):
        for k in tqdm(range(n_z)):
            if HAS_OBJECT:
                OBJECT_Z = z_arm[i][j][k]

            count += 1
            state = [x_arm[i][j][k],
                    y_arm[i][j][k],
                    z_arm[i][j][k],
                    x_dist[i][j][k],
                    y_dist[i][j][k],
                    z_dist[i][j][k],
                    OBJECT_Z,
                    HAS_THE_TARGET,
                    HAND_IS_CLOSED]

            r.endpoint[0] = x_arm[i][j][k]
            r.endpoint[1] = y_arm[i][j][k]
            r.endpoint[2] = z_arm[i][j][k]
            r.set_target_location(OBJECT_X,OBJECT_Y,OBJECT_Z)
            if not HAS_OBJECT:
                r.set_target_location(OBJECT_X,OBJECT_Y,OBJECT_Z)
            else:
                r.set_target_location(x_arm[i][j][k],y_arm[i][j][k],z_arm[i][j][k]-0.05)

            r.set_hand_close(HAND_IS_CLOSED)
            r.has_target = HAS_THE_TARGET

            target_location = r.get_target_location()
            s = simulator.Simulator(r,1,target_x=target_location[0],target_y=target_location[1],target_z=target_location[2])
            for step in range(5):
                s.step()
#            print("> visualizer: endpoint at",r.endpoint)
            img_state, numerical_state = s.get_robot_state()

            img_state_tensor = torch.Tensor(img_state).to(device)
            numerical_state_tensor = torch.Tensor(numerical_state).to(device)

            values_tensor = br.get_v_from_state(img_state_tensor.unsqueeze(0).transpose(1,3).transpose(2,3), numerical_state_tensor.unsqueeze(0)).cpu()
            best_action = br.maximize_q_network(img_state_tensor, numerical_state_tensor).cpu()

            values[i][j][k] = values_tensor.item()
            best_action_np = best_action.numpy()
            action_x[i][j][k] = best_action_np[0].item()
            action_y[i][j][k] = best_action_np[1].item()
            action_z[i][j][k] = best_action_np[2].item()

            # plot the value functions
            #ax.text(x_arm[i][j][k],
            #        y_arm[i][j][k],
            #        z_arm[i][j][k],
            #        round(values[i][j][k],2))

            # plot the termination
            if(best_action_np[-3].item() > 0):
                ax.text(x_arm[i][j][k],
                        y_arm[i][j][k],
                        z_arm[i][j][k],
                        "T")

            #plot the gripper open/close, +ve = close, -ve = open
            if (best_action_np[-2] - best_action_np[-1] > 0):
                #red = hand should close
#                pass
                ax.scatter([x_arm[i][j][k]],[y_arm[i][j][k]],[z_arm[i][j][k]],s=30,c='r')
            else:
                #green = hand should open
#                pass
                ax.scatter([x_arm[i][j][k]],[y_arm[i][j][k]],[z_arm[i][j][k]],s=30,c='g')


title_text = "HAS_OBJECT = "+str(HAS_OBJECT)
plt.title(title_text)
ax.quiver(x_arm,
        y_arm,
        z_arm,
        action_x,
        action_y,
        action_z,
        length=0.1,
        pivot='tail')

ax.scatter([0],[0],[0],s=100,c='r')
ax.scatter([OBJECT_X],[OBJECT_Y],[OBJECT_Z],s=100,c='b')

# for visualization purpose
values_viz = values - values.min()
#values_viz = values_viz / values_viz.max()
cax = ax.scatter(x_arm,
            y_arm,
            z_arm,
            values_viz,
            c = np.abs(values_viz.flatten()),
            s=100,
            cmap=plt.get_cmap("plasma"))

fig.colorbar(cax)
print("value min =",values.min())
print("value max =",values.max())
plt.show()

#================================================================================
