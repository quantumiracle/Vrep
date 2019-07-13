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
