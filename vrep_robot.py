# Import some modules
import os
import time
from subprocess import Popen
import numpy as np
import vrep

class VrepRobot:
    def __init__(self,dt):
        self.dt = dt
        self.endpoint = np.array([0,0,0])
        self.v_endpoint = np.array([0,0,0])
        self.omega = 0
        self.yaw = 0
        self.hand_is_closed = 0
        self.NUM_JOINTS = 9
        self.joint_handles = [None] * self.NUM_JOINTS
        self.arm_reach = 0.6
        self.enable_visualize = True
        self.img_h = 128
        self.img_w = 128
        self.img_channel = 1

        # ====================== connect to vrep =================================
        # Function to check for errors when calling a remote API function

        print("> vrep_robot: connecting to vrep")
        # Define the port number where communication will be made to the V-Rep server
        port_num = 19990
        # Define the host where this communication is taking place (the local machine, in this case)
        host = '127.0.0.1'

        # Launch a V-Rep server
        # Read more here: http://www.coppeliarobotics.com/helpFiles/en/commandLine.htm
        remote_api_string = '-gREMOTEAPISERVERSERVICE_' + str(port_num) + '_FALSE_TRUE'
        #args = ['vrep.sh', remote_api_string]
        vrep_path = "/home/clay/masters/vrep/V-REP_PRO_EDU_V3_6_0_Ubuntu16_04/vrep.sh"
        #vrep_path = "/homes/gt4118/Desktop/V-REP_PRO_EDU_V3_6_0_Ubuntu18_04/vrep.sh"
        parent_dir = os.path.abspath(os.path.join("..", os.pardir))
        args = [vrep_path, remote_api_string]
        self.process = Popen(args, preexec_fn=os.setsid)
        time.sleep(6)

        # Start a communication thread with V-Rep
        self.client_id = vrep.simxStart(host, port_num, True, True, 5000, 5)
        return_code = vrep.simxSynchronous(self.client_id, enable=True) #originally True
        self.check_for_errors(return_code)

        # Load the scene
        dir_path = os.path.dirname(os.path.realpath(__file__))
        scene_path = dir_path + '/ik_one.ttt'
        return_code = vrep.simxLoadScene(self.client_id, scene_path, 0, vrep.simx_opmode_blocking)
        self.check_for_errors(return_code)

        self.initialize_joint_handles()
        _, initial_pose, _, _ = self.call_lua_function('get_endpoint_position')
        _, yaw, _, _ = self.call_lua_function('get_endpoint_yaw')
        self.move_endpoint_to(initial_pose[0],initial_pose[1],initial_pose[2])
        self.initial_endpoint = np.array(initial_pose)
        self.initial_yaw = yaw[0]
        self.initial_joint_position = self.get_complete_pose()

        # Start the simulation (the "Play" button in V-Rep should now be in a "Pressed" state)
        return_code = vrep.simxStartSimulation(self.client_id, vrep.simx_opmode_oneshot_wait)
        self.check_for_errors(return_code)
        print("> vrep_robot: vrep connection done")

        # Get the initial configuration of the robot (needed to later reset the robot's pose)
        init_config_tree, _, _, _ = self.call_lua_function('get_configuration_tree', opmode=vrep.simx_opmode_blocking)
        # ======================================================================

    def __del__(self):
        # Shutdown
        vrep.simxStopSimulation(self.client_id, vrep.simx_opmode_blocking)
        vrep.simxFinish(self.client_id)
        pgrp = os.getpgid(self.process.pid)
        os.killpg(pgrp, signal.SIGINT)

    # Function to call a Lua function in V-Rep
    # Some things (such as getting the robot's joint velocities) do not have a remote (Python) API function, only a regular API function
    # Therefore, they need to be called directly in V-Rep using Lua (see the script attached to the "remote_api" dummy in the V-Rep scene)
    # Read more here: http://www.coppeliarobotics.com/helpFiles/en/remoteApiExtension.htm
    def call_lua_function(self, lua_function, ints=[], floats=[], strings=[], bytes=bytearray(), opmode=vrep.simx_opmode_blocking):
        return_code, out_ints, out_floats, out_strings, out_buffer = vrep.simxCallScriptFunction(self.client_id, 'remote_api', vrep.sim_scripttype_customizationscript, lua_function, ints, floats, strings, bytes, opmode)
#        print("return code for:",lua_function,"is:",return_code,"success code is:",vrep.simx_return_ok)
        self.check_for_errors(return_code)
        return out_ints, out_floats, out_strings, out_buffer

    def check_for_errors(self,code):
        if code == vrep.simx_return_ok:
            return
        elif code == vrep.simx_return_novalue_flag:
            # Often, not really an error, so just ignore
            pass
        elif code == vrep.simx_return_timeout_flag:
            raise RuntimeError('The function timed out (probably the network is down or too slow)')
        elif code == vrep.simx_return_illegal_opmode_flag:
            raise RuntimeError('The specified operation mode is not supported for the given function')
        elif code == vrep.simx_return_remote_error_flag:
            raise RuntimeError('The function caused an error on the server side (e.g. an invalid handle was specified)')
        elif code == vrep.simx_return_split_progress_flag:
            raise RuntimeError('The communication thread is still processing previous split command of the same type')
        elif code == vrep.simx_return_local_error_flag:
            raise RuntimeError('The function caused an error on the client side')
        elif code == vrep.simx_return_initialize_error_flag:
            raise RuntimeError('A connection to vrep has not been made yet. Have you called connect()? (Port num = ' + str(return_code.port_num))

    def step(self):
        self.endpoint = self.endpoint + self.v_endpoint*self.dt
        self.yaw = self.yaw + self.omega*self.dt
        if self.yaw > np.pi:
            self.yaw-=2*np.pi
        if self.yaw < -np.pi:
            self.yaw+=2*np.pi
        self.move_endpoint_to(self.endpoint[0],self.endpoint[1],self.endpoint[2])
        self.set_endpoint_yawrate(self.omega)
        # this code triggers the next simulation step
        vrep.simxSynchronousTrigger(self.client_id)
        vrep.simxGetPingTime(self.client_id)

    def set_endpoint_v(self,vx,vy,vz,omega):
        self.v_endpoint = np.array([vx,vy,vz])
        self.omega = omega

    def move_endpoint_to(self,px,py,pz):
        self.endpoint = np.array([px,py,pz])
        goal = [px,py,pz]
        success, _, _, _ = self.call_lua_function('set_endpoint_position', floats=goal)

    def set_endpoint_yawrate(self,omega):
        self.omega = omega
        success, _, _, _ = self.call_lua_function('set_endpoint_yawrate', floats=[omega])

    def get_endpoint_position(self):
        return self.endpoint

    def get_yaw(self):
        return self.yaw

    def set_hand_close(self,set_to_close=1):
        if set_to_close==1:
            success, _, _, _ = self.call_lua_function('set_hand_open', ints=[0])
            self.hand_is_closed = 1
        else:
            success, _, _, _ = self.call_lua_function('set_hand_open', ints=[1])
            self.hand_is_closed = 0

    def initialize_joint_handles(self):
        # get all of the joint handles
        # Get V-Rep handles for the robot's joints
        for i in range(self.NUM_JOINTS):
            return_code, handle = vrep.simxGetObjectHandle(self.client_id, 'redundantRob_joint' + str(i + 1), vrep.simx_opmode_blocking)
            self.check_for_errors(return_code)
            self.joint_handles[i] = handle

    def get_complete_pose(self):
        joint_positions = []
        for i in range(self.NUM_JOINTS):
            return_code,joint_position = vrep.simxGetJointPosition(self.client_id,self.joint_handles[i],vrep.simx_opmode_blocking)
            self.check_for_errors(return_code)
            joint_positions.append(joint_position)

        return joint_positions

    def set_complete_pose(self,joint_positions):
        if(len(joint_positions) != self.NUM_JOINTS):
            raise RuntimeError("error: vrep_robot joint command does not match with number of joints: NUM_JOINTS="+str(self.NUM_JOINTS)+", given length="+str(len(joint_positions)))
        for i in range(self.NUM_JOINTS):
            return_code = vrep.simxSetJointPosition(self.client_id,self.joint_handles[i],joint_positions[i],vrep.simx_opmode_blocking)
            self.check_for_errors(return_code)

    def set_target_location(self,px,py,pz=0.0):
        target = [px,py,pz]
        success, _, _, _ = self.call_lua_function('set_target_location', floats=target)

    def set_goal_location(self,px,py,pz=0.0):
        goal = [px,py,pz]
        success, _, _, _ = self.call_lua_function('set_target_location', floats=goal)

    def set_visualize(self,visualize):
        self.enable_visualize = visualize

    def get_image(self):
        _, img, _, _ = self.call_lua_function('get_camera_image')
        img_np = (np.array(img)*255).astype(np.uint8)
        img_np = np.reshape(img_np,(self.img_h,self.img_w,self.img_channel))
        return img_np

    def reset(self):
        # revert the robot back to original state and randomly place the target around
#        print(self.initial_joint_position)
        self.set_complete_pose(self.initial_joint_position)
        self.move_endpoint_to(self.initial_endpoint[0],self.initial_endpoint[1],self.initial_endpoint[2])
        # we also need to set the endpoint yaw back

        self.v_endpoint = np.array([0,0,0])
        self.omega = 0

        # step the simulation a bit to help stabilize the robot
        for i in range(5):
            self.step()
