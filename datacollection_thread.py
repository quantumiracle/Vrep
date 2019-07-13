import threading
import time
from itertools import count
import brain
import simulator
import dqn
import vrep_sawyer
import simulator
import valuenet
import numpy as np
import torch
import signal
import Tkinter as tk
import logging
import logger

exitFlag = 0

class DataCollectionThread(threading.Thread):
    def __init__(self, name, memory, maxtime=20, dt=0.05, port=19991, visualize=False,
                use_scripted_policy=False, epsilon=0.0, vrep_file_name='ik_sawyer.ttt'):
        print("> DCThread: launching thread")
        threading.Thread.__init__(self)
        self.printer = logger.Printer("DCThread")

        self.r = vrep_sawyer.VrepSawyer(dt,headless_mode=True,port_num=port,vrep_file_name=vrep_file_name)
        self.s = simulator.Simulator(self.r,dt,target_x=0,target_y=0,target_z=0,visualize=False)

        if torch.cuda.is_available():
            self.printer.print_to_screen("> DCThread: cuda is available :D")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.name = name
        self.memory = memory
        self.use_scripted_policy = use_scripted_policy
        self.epsilon = epsilon

        USE_VALUE_NET = False
        STATE_IS_IMAGE = True
        MEMORY_SIZE = 1000000
        self.s.set_visualize(visualize)
        policy_net = None
        target_net1 = None
        target_net2 = None
        value_net = None
        value_net_trainer = None

        # create the target and policy networks
        policy_net = dqn.DQN().to(self.device)
        policy_net.load_state_dict(torch.load('vrep_arm_model.pt'))
        self.printer.print_to_screen("> DCThread: loaded existing model file")

        target_net1 = dqn.DQN().to(self.device)
        target_net2 = dqn.DQN().to(self.device)
        value_net = valuenet.ValueNet().to(self.device)
        value_net_trainer = valuenet.ValueNetTrainer(value_net)
        print_string = "> DCThread: number of parameters: ",sum(p.numel() for p in policy_net.parameters() if p.requires_grad)
        self.printer.print_to_screen(print_string)

        target_net1.load_state_dict(policy_net.state_dict())
        target_net1.eval()
        target_net2.load_state_dict(policy_net.state_dict())
        target_net2.eval()

        self.br = brain.Brain(
            simulator=self.s, #only to access scripted policy
            policy_net=policy_net,
            target_net1=target_net1,
            target_net2=target_net2,
            memory_size=MEMORY_SIZE,
            value_net_trainer=value_net_trainer,
            state_is_image=STATE_IS_IMAGE,
            use_value_net=USE_VALUE_NET)

        self.MAX_TIME = maxtime
        self.FRAME_SKIP = 1

        # attach the current memory to the brain
        self.br.memory = self.memory

        self.current_success_rate = None
        self.current_reward = None

    def __del__(self):
        print("> DCThread: shutting down")

    def reload_policy(self):
        start_time = time.time()
        self.br.policy_net.load_state_dict(torch.load('vrep_arm_model.pt'))
        print_string = "> DCThread: reloaded the policy network, "+str(time.time()-start_time)+" sec"
        self.printer.print_to_screen(print_string)

    def run(self):
        REPORT_EVERY = 10
        RELOAD_EVERY = 100
        total_reached = 0
        total_reward = 0
        start_time = time.time()

        for i_episode in count():
            reached, reward = self.add_episode()
            total_reward += reward
            total_reached += reached

            if i_episode % RELOAD_EVERY == 0:
                self.reload_policy()

            if i_episode % REPORT_EVERY == 0:
                time_per_ep = (time.time() - start_time)*1.0 / REPORT_EVERY
                reward_per_ep = total_reward*1.0 / REPORT_EVERY
                reach_per_ep = total_reached*1.0 / REPORT_EVERY

                print_string = "> DCThread: reward per ep:{:.3f}".format(reward_per_ep)+" success rate:{:.3f}".format(reach_per_ep*100)+" time per ep:{:.2f}".format(time_per_ep)+" sec/ep"
                self.printer.print_to_screen(print_string)

                self.current_reward = reward_per_ep
                self.current_success_rate = reach_per_ep
                total_reached = 0
                total_reward = 0
                start_time = time.time()

    def get_performance(self):
        return self.current_success_rate, self.current_reward

    def add_episode(self):
        self.s.reset()
        target_x,target_y,target_z = self.s.randomly_place_target()
        img_state, numerical_state = self.s.get_robot_state()
        error = np.random.normal(0,0.1)
        error = 0
        reached = 0
        episode_reward = 0
        time_start = time.time()

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
            action = None
            if self.use_scripted_policy:
                action = self.br.select_action_scripted_exploration(thresh=1.0,error=error)
            else:
                action = self.br.select_action_epsilon_greedy(img_state,numerical_state,self.epsilon)

            self.s.set_control(action.view(-1).cpu().numpy())
            self.s.step()

            img_state, numerical_state = self.s.get_robot_state()
            reward_number,done = self.s.get_reward_and_done(numerical_state)
            reward = torch.tensor([reward_number], device=self.device)
            episode_reward += (self.br.GAMMA**t)*reward_number

            if done and reward_number > 0:
                reached += 1

            if t>self.MAX_TIME:
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
            if (t%self.FRAME_SKIP == 0):
                self.br.memory.push(
                    torch.Tensor(last_img_state),
                    torch.Tensor(last_numerical_state),
                    action.view(-1).float(),
                    state_img_tensor,
                    state_numerical_tensor,
                    reward)

            if done:
                break

        return reached, episode_reward


if __name__ == "__main__":
    # Create new threads
    memory_buffer = dqn.ReplayMemory(1000000)
    thread1 = DataCollectionThread("DCThread", memory_buffer, maxtime=20, dt=0.05, port=19991)
    # Start new Threads
    thread1.daemon = True # daemon thread exits with the main thread, which is good
    thread1.start()
    while(True):
        #print(memory_buffer.memory)
        time.sleep(0.5)
    print "Exiting Main Thread"
