import bbopt
import dqn
import valuenet

import math
import random
import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F

class Brain:
    def __init__(self, simulator, policy_net, target_net1, target_net2, memory_size,
    online_memory_size=50000, value_net_trainer=None, state_is_image=False, use_value_net=False):
        self.s = simulator
        self.policy_net = policy_net
        self.target_net1 = target_net1
        self.target_net2 = target_net2
#        self.optimizer = optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(),lr=0.0001,weight_decay=7e-5)
        self.MEMORY_SIZE = memory_size
        self.memory = dqn.ReplayMemory(memory_size)
        self.memory_online = dqn.ReplayMemory(online_memory_size)
        self.value_net_trainer = value_net_trainer

        # set the hyperparameters
        self.BATCH_SIZE = 32
        self.GAMMA = 0.9
        self.IMG_CHANNEL = 3
        self.ACTION_BOUNDS = np.array([
            [-1.0,1.0], # vx
            [-1.0,1.0], # vy
            [-1.0,1.0], # vz
        #    [-1.0,1.0], # wx
        #    [-1.0,1.0], # wy
        #    [-1.0,1.0], # wz
            [-1.0,1.0], # terminate
            [0.0,1.0], # gripper position, close
            [0.0,1.0]  # gripper position, open
        ])
        self.N_ACTIONS = self.ACTION_BOUNDS.shape[0]
        self.CEM_ITER = 2
        self.SHOULD_TRAIN_VALUE_NET = True
        self.STATE_IS_IMAGE = state_is_image
        self.USE_VALUE_NET = use_value_net

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_random_action_tensor(self,hand_actions = None):
#        random_action_array = np.random.rand(self.N_ACTIONS) * (self.ACTION_BOUNDS[:,1]-self.ACTION_BOUNDS[:,0]) + self.ACTION_BOUNDS[:,0]
        random_action_array = np.random.multivariate_normal(
                                (self.ACTION_BOUNDS[:,1]+self.ACTION_BOUNDS[:,0])/2.0,
                                np.diag(self.ACTION_BOUNDS[:,1]-self.ACTION_BOUNDS[:,0])/2.0)
        if(hand_actions is not None):
            rand_number = np.random.uniform()
            if rand_number < 0.75:
                random_action_array[-2] = hand_actions[0]
                random_action_array[-1] = hand_actions[1]

        rand_number = np.random.uniform()
        if rand_number > 0.08:
            random_action_array[-3] = abs(random_action_array[-3])
        else:
            random_action_array[-3] = -1*abs(random_action_array[-3])

        random_action_tensor = torch.Tensor(random_action_array).to(self.device)
        return random_action_tensor

    def select_action_scripted_exploration(self,thresh=None,error=0.0):
        mode = random.random()
        THRESH_FOR_03_SUCCESS = 0.7
        threshold = THRESH_FOR_03_SUCCESS
        if thresh is not None:
            threshold = thresh
        action = self.s.get_optimal_control(error)
        if mode < threshold:
            action = torch.tensor(action)
        else:
            action = self.get_random_action_tensor(hand_actions = [action[-2],action[-1]]) #add noise except the hand open or close
        return action.view(1,-1)

    def select_action_epsilon_greedy(self,img_state,numerical_state,epsilon=None):
        sample = random.random()
        # select with epsilon-greedy with e = e_threshold which is decaying from 0.9 to 0.05
        eps_threshold = 0.2
        if epsilon is not None:
            eps_threshold = epsilon

        if sample > eps_threshold:
            best_action_tensor = self.maximize_q_network(img_state,numerical_state)
            return best_action_tensor
        else:
            action = self.get_random_action_tensor()
            return action

    def maximize_q_network(self,img_state,numerical_state):
        with torch.no_grad():
            # repeat the states across all dimesions of available acitons
            img_state_tensor = img_state.unsqueeze(0).transpose(1,3).transpose(2,3)
            numerical_state_tensor = numerical_state.view(1,-1)

            # get the list of the best action to do
            bbMaximizer = bbopt.BboptMaximizer(
                            self.policy_net,
                            None,
                            self.ACTION_BOUNDS,
                            self.CEM_ITER,
                            state_img_batch=img_state_tensor,
                            state_numerical_batch=numerical_state_tensor,
                            state_is_image=self.STATE_IS_IMAGE
                        )
#            best_action = bbMaximizer.get_argmax().view(-1)
#            best_action_tensor = torch.Tensor(best_action.cpu().numpy())
            best_action_tensor = bbMaximizer.get_argmax().view(-1)

            return best_action_tensor

    def get_v_from_state(self,state_img_batch,state_numerical_batch):
        bbMaximizer = bbopt.BboptMaximizer(
                        self.target_net1,
                        None,
                        self.ACTION_BOUNDS,
                        self.CEM_ITER,
                        state_img_batch=state_img_batch,
                        state_numerical_batch=state_numerical_batch,
                        state_is_image=self.STATE_IS_IMAGE
                    )
        best_action_tensor = bbMaximizer.get_argmax()
        best_action_tensor = best_action_tensor.view(-1,self.N_ACTIONS)

        # we can then predict it with the target network
        v_tensor1 = self.target_net1(state_img_batch.to(self.device), state_numerical_batch.to(self.device), best_action_tensor.to(self.device)).view(-1)
        v_tensor2 = self.target_net2(state_img_batch.to(self.device), state_numerical_batch.to(self.device), best_action_tensor.to(self.device)).view(-1)
        values = []
        for i in range(v_tensor1.shape[0]):
            if v_tensor1[i] < v_tensor2[i]:
                values.append(v_tensor1[i])
            else:
                values.append(v_tensor2[i])

        v_tensor = torch.tensor(values)

        return v_tensor

    def sample_from_memory(self,batch_size,online=False):
        if (len(self.memory) < batch_size or len(self.memory_online) < batch_size) and online:
            if len(self.memory) < batch_size:
                print("brain: sample from memory too small, offline:",len(self.memory),"online:",len(self.memory_online))
                return None
            else:
                print("brain: sample from memory too small (online), offline:",len(self.memory),"online:",len(self.memory_online))
                return self.sample_from_memory(batch_size,online=False)

        # sample transitions from the batch, gives a list of Transition tuples
        transitions = None
        if online:
            transitions = self.memory_online.sample(batch_size)
        else:
            transitions = self.memory.sample(batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transitiget_imageon of batch-arrays.
        # zip(*transitions) = a list containing state,action,next_state,reward
        # define batch as a namedtuple containing the states, actions, next_state, reward
        # each one in the list are torch tensors
        # basically changing it from a list to a namedtuple
        batch = dqn.Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_numerical_state)), device=self.device, dtype=torch.uint8)
        non_final_next_numerical_states = None
        non_final_next_img_states = None
        no_final_next_states = False
        state_img_batch = None
        state_numerical_batch = None

        if (non_final_mask==0).all().item() == 1:
#            print("> brain: all states are terminal, this batch us useless")
            #if all of the states are terminal
            #happens during the first stages of drawing online data
            non_final_next_img_states = None
            non_final_next_numerical_states = None
            no_final_next_states = True
        else:
            non_final_next_numerical_states = torch.stack([s for s in batch.next_numerical_state if s is not None])
            if self.STATE_IS_IMAGE:
                non_final_next_img_states = torch.stack([s for s in batch.next_img_state if s is not None]).transpose(1,3).transpose(2,3)

        state_img_batch = torch.stack(batch.img_state).transpose(1,3).transpose(2,3)
        state_numerical_batch = torch.stack(batch.numerical_state)

#        print("> brain, batch.next_img_state",type(batch.next_img_state))
#        print("> brain, batch.next_img_state",batch.next_img_state)
#        print("> brain, batch.next_numerical_state",type(batch.next_numerical_state))
#        print("> brain, batch.next_numerical_state",batch.next_numerical_state)
#        print("> brain, non_final_mask",non_final_mask)
#        print("> brain, condition",((non_final_mask==0).all().item() == 1))
#        print("action batch",batch.action)

        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

#        print("state_numerical_batch",state_numerical_batch.size())
#        print("state_img_batch",state_img_batch.size())
#        print("action batch",action_batch.size())
#        print("reward batch",reward_batch.size())
#        print("next_state_numerical_batch",next_state_numerical_batch.size())
#        print("next_state_img_batch",next_state_img_batch.size())

        # action here is the [action1 action2 action3 ..]
        action_batch_reshape = action_batch.view(-1,self.N_ACTIONS)

        return state_img_batch, state_numerical_batch, action_batch_reshape, reward_batch, non_final_mask, non_final_next_img_states, non_final_next_numerical_states, no_final_next_states


    # optimize the model
    def optimize_model(self):
        offline_transitions = self.sample_from_memory(self.BATCH_SIZE/2,online=False)
        online_transitions = self.sample_from_memory(self.BATCH_SIZE/2,online=True)
        if offline_transitions is None and online_transitions is None:
            return
        state_img_batch_offline, state_numerical_batch_offline, action_batch_reshape_offline, reward_batch_offline, non_final_mask_offline, non_final_next_img_states_offline, non_final_next_numerical_states_offline, no_final_next_states_offline = offline_transitions
        state_img_batch_online, state_numerical_batch_online, action_batch_reshape_online, reward_batch_online, non_final_mask_online, non_final_next_img_states_online, non_final_next_numerical_states_online, no_final_next_states_online = online_transitions
        if no_final_next_states_online:
#            print("brain: online memory has only terminal states, sampling again from offline instead")
            online_transitions = self.sample_from_memory(self.BATCH_SIZE/2,online=False)
            state_img_batch_online, state_numerical_batch_online, action_batch_reshape_online, reward_batch_online, non_final_mask_online, non_final_next_img_states_online, non_final_next_numerical_states_online, no_final_next_states_online = online_transitions

#        print("state_img_batch_offline",state_img_batch_offline)
#        print("state_numerical_batch_offline",state_numerical_batch_offline)
#        print("action_batch_reshape_offline",action_batch_reshape_offline)
#        print("action_batch_reshape_online",action_batch_reshape_online)
#        print("reward_batch_offline",reward_batch_offline)
#        print("non_final_mask_offline",non_final_mask_offline)
#        print("non_final_next_img_states_online",non_final_next_img_states_online)
#        print("non_final_next_numerical_states_online",non_final_next_numerical_states_online)
#        print("no_final_next_states_online",no_final_next_states_online)

        state_img_batch = torch.cat([state_img_batch_offline.cpu(), state_img_batch_online.cpu()])
        state_numerical_batch = torch.cat([state_numerical_batch_offline.cpu(), state_numerical_batch_online.cpu()])
        action_batch_reshape = torch.cat([action_batch_reshape_offline.cpu(), action_batch_reshape_online.cpu()])
        reward_batch = torch.cat([reward_batch_offline.cpu(), reward_batch_online.cpu()])
        non_final_mask = torch.cat([non_final_mask_offline.cpu(), non_final_mask_online.cpu()])
        non_final_next_img_states = torch.cat([non_final_next_img_states_offline.cpu(), non_final_next_img_states_online.cpu()])
        non_final_next_numerical_states = torch.cat([non_final_next_numerical_states_offline.cpu(), non_final_next_numerical_states_online.cpu()])

        state_action_values = self.policy_net(
                                state_img_batch.to(self.device),
                                state_numerical_batch.to(self.device),
                                action_batch_reshape.to(self.device)
                                )

#        # train the value net
#        next_state_values = None
        if self.value_net_trainer.should_train_net:
            # Compute V(s_{t+1}) for all next states.
            # Expected values of actions for non_final_next_states are computed based
            # on the "older" target_net; selecting their best reward with max(1)[0].
            # This is merged based on the mask, such that we'll have either the expected
            # state value or 0 in case the state was final.
            # V(s) = max_a(Q(s,a))
            next_state_values = torch.zeros(self.BATCH_SIZE)
            next_state_values[non_final_mask] = self.get_v_from_state(non_final_next_img_states, non_final_next_numerical_states).detach()
#            if self.USE_VALUE_NET:
#                print("Trainer: training value net")
#                self.value_net_trainer.set_data(non_final_next_states)
#                self.value_net_trainer.set_labels(next_state_values)
#                self.value_net_trainer.train()
#
#        if self.USE_VALUE_NET:
#            value_net_error = self.value_net_trainer.evaluate()
#            #print("value net error is",value_net_error)
#            # if the error of the value net is below 0.5%
#            if abs(value_net_error) < 0.5:
#                if self.value_net_trainer.should_train_net:
#                    # print only once
#                    print("> Trainer: valuenet error =",value_net_error,"diabling target network for this episode")
#                #let's just use the value net for this episode
#                #if we already calculate it the accurate and hard way, just use it, it's better
#                self.value_net_trainer.disable_training()
#                if next_state_values is None:
#                    next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
#                    # somehow the output is in shape (batchsize,1)
#                    next_state_values[non_final_mask] = self.value_net_trainer.model(non_final_next_states).reshape(-1).detach()
#            else:
#                print("> Trainer: valuenet error =",value_net_error,"not quite there yet")

        # Compute the expected Q values (this is the predicted action value based on the policy network)
#        print("next_state_values",next_state_values.device)
#        print("reward_batch",reward_batch.device)
        expected_state_action_values = (next_state_values.to(self.device) * self.GAMMA) + reward_batch.to(self.device)

        # Compute loss
#        loss = F.binary_cross_entropy(state_action_values, expected_state_action_values.unsqueeze(1))
        loss = F.mse_loss(state_action_values.to(self.device), expected_state_action_values.to(self.device).unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            # somehow some weights in some layers does not have gradients, i think it is the pooling layers
            if(param.grad is not None):
#                print("gradients:",param.grad.data)
                param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        #copy the weights of the policy net to target net 1
        for target_net_param, policy_net_param in zip(self.target_net1.parameters(), self.policy_net.parameters()):
            target_net_param.data.copy_((1-0.9999) * policy_net_param.data + target_net_param.data * 0.9999)

        return loss.cpu().item()

    def update_target_net(self):
        self.target_net2.load_state_dict(self.target_net1.state_dict())
'''
Soft Actor-Critic version 2
using target Q instead of V net: 2 Q net, 2 target Q net, 1 policy net
add alpha loss compared with version 1
paper: https://arxiv.org/pdf/1812.05905.pdf
'''


import math
import random

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

from IPython.display import clear_output
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import display
import argparse
import gzip

import vrep_sawyer
import simulator
import tqdm
import bbopt
import sys

if sys.version_info[0] < 3:
    print("Python2")
    import cPickle
else:
    print("Python3")
    import pickle as cPickle

GPU = True
device_idx = 0
if GPU:
    device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
print(device)


parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--train', dest='train', action='store_true', default=False)
parser.add_argument('--test', dest='test', action='store_true', default=False)

args = parser.parse_args()


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = int((self.position + 1) % self.capacity)  # as a ring buffer
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch)) # stack for each element
        ''' 
        the * serves as unpack: sum(a,b) <=> batch=(a,b), sum(*batch) ;
        zip: a=[1,2], b=[2,3], zip(a,b) => [(1, 2), (2, 3)] ;
        the map serves as mapping the function on each list element: map(square, [2,3]) => [4,9] ;
        np.stack((1,2)) => array([1, 2])
        '''
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

class NormalizedActions(gym.ActionWrapper):
    def _action(self, action):
        low  = self.action_space.low
        high = self.action_space.high
        
        action = low + (action + 1.0) * 0.5 * (high - low)
        action = np.clip(action, low, high)
        
        return action

    def _reverse_action(self, action):
        low  = self.action_space.low
        high = self.action_space.high
        
        action = 2 * (action - low) / (high - low) - 1
        action = np.clip(action, low, high)
        
        return action

def plot(rewards):
    clear_output(True)
    plt.figure(figsize=(20,5))
    plt.plot(rewards)
    plt.savefig('sac_v2.png')
    # plt.show()

class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, init_w=3e-3):
        super(ValueNetwork, self).__init__()
        
        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, 1)
        # weights initialization
        self.linear4.weight.data.uniform_(-init_w, init_w)
        self.linear4.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return x
        
        
class SoftQNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3):
        super(SoftQNetwork, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, 1)
        
        self.linear4.weight.data.uniform_(-init_w, init_w)
        self.linear4.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state, action):
        x = torch.cat([state, action], 1) # the dim 0 is number of samples
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return x
        
        
class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, action_range=1., init_w=3e-3, log_std_min=-20, log_std_max=2):
        super(PolicyNetwork, self).__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, hidden_size)

        self.mean_linear = nn.Linear(hidden_size, num_actions)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)
        
        self.log_std_linear = nn.Linear(hidden_size, num_actions)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)

        self.action_range = action_range
        self.num_actions = num_actions

        
    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))

        mean    = (self.mean_linear(x))
        # mean    = F.leaky_relu(self.mean_linear(x))
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def evaluate(self, state, epsilon=1e-6):
        '''
        generate sampled action with state as input wrt the policy network;
        '''
        mean, log_std = self.forward(state)
        std = log_std.exp() # no clip in evaluation, clip affects gradients flow
        
        normal = Normal(0, 1)
        z      = normal.sample() 
        action_0 = torch.tanh(mean + std*z.to(device)) # TanhNormal distribution as actions; reparameterization trick
        action = self.action_range*action_0
        log_prob = Normal(mean, std).log_prob(mean+ std*z.to(device)) - torch.log(1. - action_0.pow(2) + epsilon) -  np.log(self.action_range)
        # both dims of normal.log_prob and -log(1-a**2) are (N,dim_of_action); 
        # the Normal.log_prob outputs the same dim of input features instead of 1 dim probability, 
        # needs sum up across the features dim to get 1 dim prob; or else use Multivariate Normal.
        log_prob = log_prob.sum(dim=1, keepdim=True)
        return action, log_prob, z, mean, log_std
        
    
    def get_action(self, state, deterministic):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        normal = Normal(0, 1)
        z      = normal.sample().to(device)
        action = self.action_range* torch.tanh(mean + std*z)
        
        action = self.action_range*mean.detach().cpu().numpy()[0] if deterministic else action.detach().cpu().numpy()[0]
        return action


    def sample_action(self,):
        a=torch.FloatTensor(self.num_actions).uniform_(-1, 1)
        return self.action_range*a.numpy()


class SAC_Trainer():
    def __init__(self, replay_buffer, hidden_dim, action_range):
        self.replay_buffer = replay_buffer

        self.soft_q_net1 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.soft_q_net2 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_soft_q_net1 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_soft_q_net2 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim, action_range).to(device)
        self.log_alpha = torch.zeros(1, dtype=torch.float32, requires_grad=True, device=device)
        print('Soft Q Network (1,2): ', self.soft_q_net1)
        print('Policy Network: ', self.policy_net)

        for target_param, param in zip(self.target_soft_q_net1.parameters(), self.soft_q_net1.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_soft_q_net2.parameters(), self.soft_q_net2.parameters()):
            target_param.data.copy_(param.data)

        self.soft_q_criterion1 = nn.MSELoss()
        self.soft_q_criterion2 = nn.MSELoss()

        soft_q_lr = 3e-4
        policy_lr = 3e-4
        alpha_lr  = 3e-4

        self.soft_q_optimizer1 = optim.Adam(self.soft_q_net1.parameters(), lr=soft_q_lr)
        self.soft_q_optimizer2 = optim.Adam(self.soft_q_net2.parameters(), lr=soft_q_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)

    
    def update(self, batch_size, reward_scale=10., use_demonstration=False, auto_entropy=True, target_entropy=-2, gamma=0.99,soft_tau=1e-2):
        state, action, reward, next_state, done = self.replay_buffer.sample(int(batch_size/2))
        # print('sample:', state, action,  reward, done)  # 2D: state, action  1D: reward, done


        state      = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        action     = torch.FloatTensor(action).to(device)
        reward     = torch.FloatTensor(reward).unsqueeze(1).to(device)  # reward is single value, unsqueeze() to add one dim to be [reward] at the sample dim;
        done       = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

        if use_demonstration: 
            # sample from demonstration data
            with gzip.open('dataset/replay_02.gz', 'rb') as handle:
                if sys.version_info[0] < 3:
                    demonstration_data = cPickle.load(handle)
                    transitions = demonstration_data.sample(int(batch_size/2)) 
                    for i in range(len(transitions)):
                        if transitions[i].next_numerical_state is not None:
                            state = torch.cat((state, transitions[i].numerical_state.unsqueeze(0).to(device)), 0)
                            next_state = torch.cat((next_state, transitions[i].next_numerical_state.unsqueeze(0).to(device)), 0)
                            action = torch.cat((action, transitions[i].action.unsqueeze(0).to(device)), 0)
                            reward = torch.cat((reward, transitions[i].reward.unsqueeze(1).to(device)), 0)
                            done = torch.cat((done, torch.FloatTensor([[1]]).to(device)), 0)
                else:
                    demonstration_data = cPickle.load(handle, encoding='bytes')
                # print(demonstration_data.__dict__)
                # after using bytes encoding, needs to decode with b'key' on dictionary
                # print('second: ', demonstration_data.__dict__[b'memory'][10].numerical_state)  
            for i in range(int(batch_size/2)):
                idx=np.random.randint(0,len(demonstration_data.__dict__[b'memory']))
                if demonstration_data.__dict__[b'memory'][idx].next_numerical_state is not None:
                    state = torch.cat((state, demonstration_data.__dict__[b'memory'][idx].numerical_state.unsqueeze(0).to(device)), 0)
                    next_state = torch.cat((next_state, demonstration_data.__dict__[b'memory'][idx].next_numerical_state.unsqueeze(0).to(device)), 0)
                    action = torch.cat((action, demonstration_data.__dict__[b'memory'][idx].action.unsqueeze(0).to(device)), 0)
                    reward = torch.cat((reward, demonstration_data.__dict__[b'memory'][idx].reward.unsqueeze(1).to(device)), 0)
                    done = torch.cat((done, torch.FloatTensor([[1]]).to(device)), 0)

            # print('sample:', state, action, next_state,  reward, done)  # 2D: state, action, next_state  1D: reward, done




        predicted_q_value1 = self.soft_q_net1(state, action)
        predicted_q_value2 = self.soft_q_net2(state, action)
        new_action, log_prob, z, mean, log_std = self.policy_net.evaluate(state)
        new_next_action, next_log_prob, _, _, _ = self.policy_net.evaluate(next_state)

        reward = reward_scale * (reward - reward.mean(dim=0)) /(reward.std(dim=0) + 1e-6)# normalize with batch mean and std
    # Updating alpha wrt entropy
        # alpha = 0.0  # trade-off between exploration (max entropy) and exploitation (max Q) 
        if auto_entropy is True:
            alpha_loss = -(self.log_alpha * (log_prob + target_entropy).detach()).mean()
            # print('alpha loss: ',alpha_loss)
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = 1.
            alpha_loss = 0

    # Training Q Function
        target_q_min = torch.min(self.target_soft_q_net1(next_state, new_next_action),self.target_soft_q_net2(next_state, new_next_action)) - self.alpha * next_log_prob
        target_q_value = reward + (1 - done) * gamma * target_q_min # if done==1, only reward
        q_value_loss1 = self.soft_q_criterion1(predicted_q_value1, target_q_value.detach())  # detach: no gradients for the variable
        q_value_loss2 = self.soft_q_criterion2(predicted_q_value2, target_q_value.detach())


        self.soft_q_optimizer1.zero_grad()
        q_value_loss1.backward()
        self.soft_q_optimizer1.step()
        self.soft_q_optimizer2.zero_grad()
        q_value_loss2.backward()
        self.soft_q_optimizer2.step()  

    # Training Policy Function
        predicted_new_q_value = torch.min(self.soft_q_net1(state, new_action),self.soft_q_net2(state, new_action))
        policy_loss = (self.alpha * log_prob - predicted_new_q_value).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        # print('q loss: ', q_value_loss1, q_value_loss2)
        # print('policy loss: ', policy_loss )


    # Soft update the target value net
        for target_param, param in zip(self.target_soft_q_net1.parameters(), self.soft_q_net1.parameters()):
            target_param.data.copy_(  # copy data value into target parameters
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )
        for target_param, param in zip(self.target_soft_q_net2.parameters(), self.soft_q_net2.parameters()):
            target_param.data.copy_(  # copy data value into target parameters
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )
        return predicted_new_q_value.mean()

    def save_model(self, path):
        torch.save({'soft_q_net1': self.soft_q_net1.state_dict(), 
        'soft_q_net2': self.soft_q_net2.state_dict(),
        'target_soft_q_net1': self.target_soft_q_net1.state_dict(),
        'target_soft_q_net2': self.target_soft_q_net2.state_dict(),
        'policy_net': self.policy_net.state_dict()
        }, 
        path)
        # save log-alpha variable
        torch.save(self.log_alpha, path+'logalpha.pt')

    def load_model(self, path):
        checkpoint = torch.load(path)
        self.soft_q_net1.load_state_dict(checkpoint['soft_q_net1'])
        self.soft_q_net2.load_state_dict(checkpoint['soft_q_net2'])
        self.target_soft_q_net1.load_state_dict(checkpoint['target_soft_q_net1'])
        self.target_soft_q_net2.load_state_dict(checkpoint['target_soft_q_net2'])
        self.policy_net.load_state_dict(checkpoint['policy_net'])

        self.log_alpha = torch.load(path+'logalpha.pt')
        # for inference
        self.soft_q_net1.eval()
        self.soft_q_net2.eval()
        self.target_soft_q_net1.eval()
        self.target_soft_q_net2.eval()
        self.policy_net.eval()

        # for re-training
        # self.soft_q_net1.train()
        # self.soft_q_net2.train()
        # self.target_soft_q_net1.train()
        # self.target_soft_q_net2.train()
        # self.policy_net.train()

if __name__ == '__main__':
    replay_buffer_size = 1e6
    replay_buffer = ReplayBuffer(replay_buffer_size)

    # choose env
    model_path = './model/sac_model'
    dt = 100e-3
    r = vrep_sawyer.VrepSawyer(dt)
    env = simulator.Simulator(r,dt,target_x=0,target_y=0,target_z=0,visualize=False)
    action_dim = 6
    state_dim = 9

    # hyper-parameters for RL training
    max_episodes  = 20000
    max_steps   = 20   # Pendulum needs 150 steps per episode to learn well, cannot handle 20
    batch_size  = 10
    explore_eps = 2  # for random action sampling in the beginning of training
    update_itr = 1
    AUTO_ENTROPY=True
    DETERMINISTIC=False
    USE_DEMONSTRATION=False
    hidden_dim = 512
    rewards     = [0]
    predict_qs  = []
    action_range = 0.2


    sac_trainer=SAC_Trainer(replay_buffer, hidden_dim=hidden_dim, action_range=action_range  )
    
    if args.train:
        sac_trainer.load_model( model_path)

        # training loop
        for ep in range(max_episodes):
            env.reset()
            target_x,target_y,target_z = env.randomly_place_target()
            vs, s = env.get_robot_state()  # fisrt dim is visual, second dim is numerical
            episode_reward = 0
            predict_q = 0
            
            
            for step in range(max_steps):
                if ep > explore_eps:
                    a = sac_trainer.policy_net.get_action(s, deterministic = DETERMINISTIC)
                else:
                    a = sac_trainer.policy_net.sample_action()
                env.set_control(a)
                env.step()
                vs_, s_ = env.get_robot_state()
                r,done = env.get_reward_and_done(s_)   
                # print('action: ', a)
                replay_buffer.push(s, a, r, s_, done)
                
                s = s_
                episode_reward += r
                
                
                if len(replay_buffer) > batch_size:
                    for i in range(update_itr):
                        predict_q=sac_trainer.update(batch_size, reward_scale=10., use_demonstration = USE_DEMONSTRATION, auto_entropy=AUTO_ENTROPY, target_entropy=-1.*action_dim)
    
                if done:
                    break
                
            if ep % 10 == 0:
                plot(rewards)
                sac_trainer.save_model( model_path)
                
            print('Episode: ', ep, '| Episode Reward: ', episode_reward, '| Episode Length: ', step)
            # plot the running mean instead of raw values
            rewards.append(0.95*np.mean(rewards)+0.05*episode_reward)
        predict_qs.append(predict_q)
    

    if args.test:
        sac_trainer.load_model( model_path)

        for ep in range(max_episodes):
            env.reset()
            target_x,target_y,target_z = env.randomly_place_target()
            vs, s = env.get_robot_state()  # fisrt dim is visual, second dim is numerical
            episode_reward = 0
            predict_q = 0
            
            
            for step in range(max_steps):
                # if frame_idx > explore_steps:
                #     a = sac_trainer.policy_net.get_action(s, deterministic = DETERMINISTIC)
                # else:
                a = sac_trainer.policy_net.sample_action()
                env.set_control(a)
                env.step()
                vs_, s_ = env.get_robot_state()
                r,done = env.get_reward_and_done(s_)   
                    
                # replay_buffer.push(s, a, r, s_, done)
                
                s = s_
                episode_reward += r
                
                
                # if len(replay_buffer) > batch_size:
                #     for i in range(update_itr):
                #         predict_q=sac_trainer.update(batch_size, reward_scale=10., auto_entropy=AUTO_ENTROPY, target_entropy=-1.*action_dim)
                
                # if frame_idx % 100 == 0:
                #     plot(frame_idx, rewards, predict_qs)
                    # sac_trainer.save_model( model_path)
                
                if done:
                    break

                
            print('Episode: ', ep, '| Episode Reward: ', episode_reward)
            rewards.append(episode_reward)
        predict_qs.append(predict_q)
