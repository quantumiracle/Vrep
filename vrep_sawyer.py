# Import some modules
import os
import time
from subprocess import Popen
import numpy as np
import vrep
import cv2
# import config
import signal

class VrepSawyer:
    def __init__(self,dt, headless_mode=True,port_num=19990,vrep_file_name=None):
        self.dt = dt
        self.endpoint = np.array([0,0,0])
        self.v_endpoint = np.array([0,0,0])
        self.omega = 0
        self.yaw = 0 # yaw is the act of the hand tip rotating like a drill
        self.pitch = 0 # pitch is the act of the hand tip pointing up and down
        self.roll = 0  # roll the the act of the hand tip pointing left and right
        self.omega_pitch = 0 # roll and pitch will not be aynchronous with vrep, the initial position is 0 roll 0 pitch
        self.omega_roll = 0
        self.hand_is_closed = 0
        self.HAND_CLOSE_THRESHOLD = 60.0/180.0*np.pi
        self.N_ARM_JOINTS = 8
        self.N_HAND_FINGERS = 5
        self.N_FINGER_JOINTS = 3
        self.NUM_JOINTS = self.N_ARM_JOINTS + self.N_HAND_FINGERS*self.N_FINGER_JOINTS
        self.joint_handles = [None] * self.NUM_JOINTS
        self.arm_reach = 1.0
        self.enable_visualize = True
        self.img_h = 64
        self.img_w = 64
        self.img_channel = 3
        self.has_target = False
        self.holding_target = False
        self.terminate_episode = -1

        self.TRAY_X = 0.8
        self.TRAY_Y = 0
        self.TRAY_DX = 0.08
        self.TRAY_DY = 0.4

        # ====================== connect to vrep =================================
        # Function to check for errors when calling a remote API function

        print("> vrep_robot: connecting to vrep")
        # Define the port number where communication will be made to the V-Rep server
        port_num = port_num
        # Define the host where this communication is taking place (the local machine, in this case)
        host = '127.0.0.1'

        # Launche a V-Rep server
        # Read more here: http://www.coppeliarobotics.com/helpFiles/en/commandLine.htm
        remote_api_string = '-gREMOTEAPISERVERSERVICE_' + str(port_num) + '_FALSE_TRUE'
        headless_string = '-h'
        quit_when_finish_string = '-q'
        disable_gui_string = 'gGUIITEMS_65536'
        #args = ['vrep.sh', remote_api_string]
        # vrep_path = ""
        # if config.is_laptop:
        #     vrep_path = "/home/clay/masters/V-REP_PRO_EDU_V3_6_1_Ubuntu18_04/vrep.sh"
        # else:
        #     vrep_path = "/homes/sc6918/masters/V-REP_PRO_EDU_V3_6_1_Ubuntu18_04/vrep.sh"
        # vrep_path = "/homes/zd2418/Software/V-REP_PLAYER_V3_6_1_Ubuntu18_04/vrep.sh"
        vrep_path = "/home/quantumiracle/Software/v-rep3.6.2/V-REP_PRO_EDU_V3_6_2_Ubuntu18_04/vrep.sh"
        parent_dir = os.path.abspath(os.path.join("..", os.pardir))
        args = [vrep_path, remote_api_string, quit_when_finish_string]
        if headless_mode:
            args.append(headless_string)
            args.append(disable_gui_string)
        self.process = Popen(args, preexec_fn=os.setsid)
        time.sleep(6)

        # Start a communication thread with V-Rep
        self.client_id = vrep.simxStart(host, port_num, True, True, 5000, 5)
        return_code = vrep.simxSynchronous(self.client_id, enable=True) #originally True
        self.check_for_errors(return_code)

        # Load the scene
        dir_path = os.path.dirname(os.path.realpath(__file__))
        if vrep_file_name is None:
            scene_path = dir_path[:dir_path.rfind('/')] + '/ik_sawyer.ttt'
        else:
            scene_path = dir_path[:dir_path.rfind('/')] + '/'+str(vrep_file_name)
        print("> vrep_sawyer: scene_path:",scene_path)
        return_code = vrep.simxLoadScene(self.client_id, scene_path, 0, vrep.simx_opmode_blocking)
        self.check_for_errors(return_code)

        self.initialize_joint_handles()
        _, initial_pose, _, _ = self.call_lua_function('get_endpoint_position')
        _, self.initial_orientation, _, _= self.call_lua_function('get_endpoint_orientation')
        self.move_endpoint_to(initial_pose[0],initial_pose[1],initial_pose[2])
        self.initial_endpoint = np.array(initial_pose)
        self.initial_endpoint_range = np.array([self.TRAY_DX/2.0, self.TRAY_DY/2, 0.1])
        self.box_center_position = np.array([self.TRAY_X, self.TRAY_Y, self.initial_endpoint[2]])
        self.initial_joint_position = self.get_complete_pose()
        self.hand_position_beginning_episode = self.get_hand_position()

        # Start the simulation (the "Play" button in V-Rep should now be in a "Pressed" state)
        return_code = vrep.simxStartSimulation(self.client_id, vrep.simx_opmode_oneshot_wait)
        self.check_for_errors(return_code)
        print("> vrep_robot: vrep connection done")

        # Get the initial configuration of the robot (needed to later reset the robot's pose)
        init_config_tree, _, _, _ = self.call_lua_function('get_configuration_tree', opmode=vrep.simx_opmode_blocking)
        # ======================================================================

    def __del__(self):
        # Shutdown
        print("sawyer: closing command called")
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
#        self.yaw = self.yaw + self.omega*self.dt
        droll = self.omega_roll*self.dt
        dpitch = self.omega_pitch*self.dt
        dyaw = self.omega*self.dt
        self.manage_angle_wraparound()
        self.move_endpoint_to(self.endpoint[0],self.endpoint[1],self.endpoint[2])
        self.increment_endpoint_angles([droll,dpitch,dyaw])
        self.update_hand_status()
        self.update_has_the_target()
        self.update_holding_target()
        # this code triggers the next simulation step
        vrep.simxSynchronousTrigger(self.client_id)
        vrep.simxGetPingTime(self.client_id)

    def manage_angle_wraparound(self):
        if self.yaw > np.pi:
            self.yaw-=2*np.pi
        if self.yaw < -np.pi:
            self.yaw+=2*np.pi
        # for roll and pitch, clipping seems to be more reasonable than wraparound
        if self.roll > np.pi:
            self.roll -= 2*np.pi
        if self.roll < -np.pi:
            self.roll += 2*np.pi
        if self.pitch > np.pi:
            self.pitch -= 2*np.pi
        if self.pitch < -np.pi:
            self.pitch += 2*np.pi

    def set_endpoint_v(self,vx,vy,vz,omega,rollrate,pitchrate):
        self.v_endpoint = np.array([vx,vy,vz])
        self.omega = omega
        self.omega_roll = rollrate
        self.omega_pitch = pitchrate

    def move_endpoint_to(self,px,py,pz):
        self.endpoint = np.array([px,py,pz])
        goal = [px,py,pz]
        success, _, _, _ = self.call_lua_function('set_endpoint_position', floats=goal)

    # incrementing the angles gives more intuitive control (for vrep)
    def increment_endpoint_angles(self,dtheta):
        self.roll += dtheta[2]
        self.pitch += dtheta[1]
        self.yaw += dtheta[0]
        success, _, _, _ = self.call_lua_function('increment_endpoint_orientation', floats=dtheta)

    def get_endpoint_orientation(self):
        return np.array([self.yaw,self.pitch,self.roll])

    def get_endpoint_position(self):
        return self.endpoint

    def get_yaw(self):
        return self.yaw

    def set_hand_close(self,set_to_close=1):
        # 1 to close the hand
        if set_to_close==1:
            success, _, _, _ = self.call_lua_function('set_hand_open', ints=[0])
        else:
            success, _, _, _ = self.call_lua_function('set_hand_open', ints=[1])

    def initialize_joint_handles(self):
        # get all of the joint handles
        # Get V-Rep handles for the robot's joints
        j = 0
        for i in range(self.N_ARM_JOINTS):
            return_code, handle = vrep.simxGetObjectHandle(self.client_id, 'Sawyer_joint' + str(i + 1), vrep.simx_opmode_blocking)
            self.check_for_errors(return_code)
            self.joint_handles[j] = handle
            j+=1
        for i_finger in range(self.N_HAND_FINGERS):
            for i_finger_joint in range(self.N_FINGER_JOINTS):
                return_code, handle = vrep.simxGetObjectHandle(self.client_id, 'finger'+str(i_finger+1)+'_joint' + str(i_finger_joint+1), vrep.simx_opmode_blocking)
                self.check_for_errors(return_code)
                self.joint_handles[j] = handle
                j+=1
        print("vrep_sawyer joint handles:",self.joint_handles)

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
        j = 0
        for i in range(self.N_ARM_JOINTS):
            return_code = vrep.simxSetJointPosition(self.client_id,self.joint_handles[i],joint_positions[i],vrep.simx_opmode_blocking)
            self.check_for_errors(return_code)
            j+=1
        for i_finger in range(self.N_HAND_FINGERS):
            for i_finger_joint in range(self.N_FINGER_JOINTS):
                return_code = vrep.simxSetJointTargetVelocity(self.client_id, self.joint_handles[j], -99, vrep.simx_opmode_blocking)
                self.check_for_errors(return_code)
                j+=1

    def set_target_location(self,px,py,pz=0.0):
        target = [px,py,pz]
        success, _, _, _ = self.call_lua_function('set_target_location', floats=target)

    def set_target_orientation(self,roll,pitch,yaw):
        #actually we can set the roll and pitch, but for a cube it will fall to 0 anyway
        success, _, _, _ = self.call_lua_function('set_target_orientation', floats=[roll,pitch,yaw])

    def get_target_location(self):
        #actually we can set the roll and pitch, but for a cube it will fall to 0 anyway
        _, locations, _, _ = self.call_lua_function('get_target_location')
        return locations

    def get_target_orientation(self):
        #actually we can set the roll and pitch, but for a cube it will fall to 0 anyway
        _, orientations, _, _ = self.call_lua_function('get_target_orientation')
        return orientations

    def set_goal_location(self,px,py,pz=0.0):
        goal = [px,py,pz]
        success, _, _, _ = self.call_lua_function('set_target_location', floats=goal)

    def set_visualize(self,visualize):
        self.enable_visualize = visualize

    def get_img(self):
        _, img, _, _ = self.call_lua_function('get_camera_image')
        img_np = (np.array(img)*255).astype(np.uint8)
        img_np = np.reshape(img_np,(self.img_h,self.img_w,self.img_channel))
#        img_np = cv2.flip(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB), 0) #image originally comes in flipped vertically, so we'll unflip it
        img_np[:,:,:] = img_np[::-1,:,:] #flip the image
        img_np[:,:,:] = img_np[:,:,::-1] #switching bgr to rgb
#        img_np = np.flip(img_np,0)
#        print('shape of retrieved image', img_np.shape)
#        print(img_np)

        # winname = "rgb_image"
        # cv2.namedWindow(winname)
        # cv2.imshow(winname,img_np)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     cv2.destroyAllWindows()

        return img_np

    def get_depth(self):
        _, img, _, _ = self.call_lua_function('get_camera_depth')
        img_np = (np.array(img)*255).astype(np.uint8)
        img_np = np.reshape(img_np,(self.img_h,self.img_w))
#        img_np = cv2.flip(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB), 0) #image originally comes in flipped vertically, so we'll unflip it
        img_np[:,:] = img_np[::-1,:] #flip the image

        img_np_visualize = (img_np-np.min(img_np))/(float(np.max(img_np)-np.min(img_np)))*255
        imC = cv2.applyColorMap(img_np_visualize.astype(np.uint8), cv2.COLORMAP_JET)

        winname = "depth_image"
        cv2.namedWindow(winname)
        cv2.moveWindow(winname, self.img_h, self.img_w)
        cv2.imshow(winname,imC)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()

        return img_np

    def get_hand_position(self):
        _, finger_positions, _, _ = self.call_lua_function('get_hand_position')
        return np.array(finger_positions)

    def update_hand_status(self):
        # the finger positions always start from 0, but some of them increases to pi/2, some of them decrease to -pi/2
        current_finger_positions = self.get_hand_position()
        difference_from_beginning = np.absolute(current_finger_positions - self.hand_position_beginning_episode)
        average_difference = np.mean(difference_from_beginning)

#        print("vrep: beginning_finger_positions",self.hand_position_beginning_episode)
#        print("vrep: current_finger_positions",current_finger_positions)
#        print("vrep: avg difference",average_difference)

        if(average_difference > self.HAND_CLOSE_THRESHOLD):
            self.hand_is_closed = True
        else:
            self.hand_is_closed = False

    def update_has_the_target(self):
        is_holding, _, _, _ = self.call_lua_function('has_target')
        self.has_target = (is_holding[0] == 1)

    def has_the_target(self):
        return self.has_target

    def update_holding_target(self):
        self.holding_target = (self.has_target and self.hand_is_closed)

    def holding_the_target(self):
        return self.holding_target

    def set_terminate_episode(self,terminate):
        self.terminate_episode = terminate

    def reset_endpoint_orientation(self):
        success, _, _, _ = self.call_lua_function('set_endpoint_orientation_wrt_sawyer', floats=self.initial_orientation)

    def reset(self, shuffle = True):
        # revert the robot back to original state and randomly place the target around
        # print(self.initial_joint_position)
        location = self.initial_endpoint
        # print('vrep, location',location)
        location_range = np.zeros(3)
        if shuffle:
            location_range = self.initial_endpoint_range
            location = np.random.uniform(self.box_center_position-location_range,self.box_center_position+location_range)

        self.set_complete_pose(self.initial_joint_position)
        self.move_endpoint_to( location[0], location[1], location[2])
        self.set_terminate_episode(-1)
        self.reset_endpoint_orientation()
        # we also need to set the endpoint yaw back

        self.v_endpoint = np.array([0,0,0])
        self.omega = 0
        self.omega_roll = 0
        self.omega_pitch = 0
        self.roll = 0
        self.pitch = 0

        # step the simulation a bit to help stabilize the robot
        for i in range(5):
            self.step()

        self.hand_position_beginning_episode = self.get_hand_position()

    # =========================================================================
    # code below is for sim2sim
    # =========================================================================

    def spawn_object_at(self, object_path, texture_path, color=[1.0,1.0,1.0], location=[0,0,0], orientation=[0,0,0]):
#        print("> vrep_sawyer, spawning object")
        success, obj_handle, _, _ = self.call_lua_function('spawn_object_to',
                                                    floats=[location[0],location[1],location[2],orientation[0],orientation[1],orientation[2],color[0],color[1],color[2]],
                                                    strings=[object_path,texture_path])
        return obj_handle[0]

    def recolor_plane(self, texture_path, color=[0.0,0.0,0.0]):
        success, _, _, _ = self.call_lua_function('set_plane_texture',
                                                    floats=[color[0],color[1],color[2]],
                                                    strings=[texture_path])

    def recolor_tray(self, texture_path, color=[0.0,0.0,0.0]):
        success, _, _, _ = self.call_lua_function('set_tray_texture',
                                                    floats=[color[0],color[1],color[2]],
                                                    strings=[texture_path])

    def recolor_head(self, texture_path, color=[0.0,0.0,0.0]):
        success, _, _, _ = self.call_lua_function('set_head_texture',
                                                    floats=[color[0],color[1],color[2]],
                                                    strings=[texture_path])

    def recolor_hand(self, texture_path, color=[0.0,0.0,0.0]):
        success, _, _, _ = self.call_lua_function('set_hand_texture',
                                                    floats=[color[0],color[1],color[2]],
                                                    strings=[texture_path])

    def recolor_arm(self, link_index, texture_path, color=[0.0,0.0,0.0]):
        success, _, _, _ = self.call_lua_function('set_sawyer_texture',
                                                    ints = [int(link_index)],
                                                    floats=[color[0],color[1],color[2]],
                                                    strings=[texture_path])

    def recolor_object(self, handle, texture_path, color=[0.0,0.0,0.0]):
        success, _, _, _ = self.call_lua_function('set_object_texture',
                                                    ints=[int(handle)],
                                                    floats=[color[0],color[1],color[2]],
                                                    strings=[texture_path])

    def remove_object(self,handle):
        success, _, _, _ = self.call_lua_function('remove_object', ints=[int(handle)])
