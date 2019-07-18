'''
wrap the vrep-sawyer environment to be a gym one
'''
import os
import time
import numpy as np

import vrep_sawyer
import simulator

class Sawyer:
    def __init__(self, dt = 100e-3, headless=False):
        r = vrep_sawyer.VrepSawyer(dt, headless_mode=headless)
        self.action_space = np.zeros(6)
        self.observation_space = np.zeros(9)
        self.simulator = simulator.Simulator(r,dt,target_x=0,target_y=0,target_z=0,visualize=True)

    def reset(self,):
        self.simulator.reset()
        target_x,target_y,target_z = self.simulator.randomly_place_target()
        vs, s = self.simulator.get_robot_state()  # fisrt dim is visual, second dim is numerical

        return vs, s


    def step(self,action):
        self.simulator.set_control(action)
        self.simulator.step()
        vs, s = self.simulator.get_robot_state()
        r,done = self.simulator.get_reward_and_done(s)   

        return vs,s,r,done


        
