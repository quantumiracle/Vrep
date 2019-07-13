import numpy as np
import cv2

class TwoDRobot:
    def __init__(self,dt):
        self.initial_endpoint = np.array([0.8,0,0.2])
        self.initial_endpoint_range = np.array([0.2,0,0.1])

        self.dt = dt
        self.x = 0
        self.z = 0.8
        self.target_x = 0
        self.target_z = 0
        self.vx = 0
        self.vz = 0
        self.has_target = False
        self.enable_visualize = True
        self.arm_reach = 1.0
        self.grasp = 0
        self.hand_is_closed = 0
        self.terminate_episode = -1

        self.SNAP_THRESH = 0.02
        self.IMAGE_W = 64
        self.IMAGE_H = 64
        self.AREA_W = 2
        self.AREA_H = 0.35
        self.G = 9.81

    def step(self):
        self.x += self.vx*self.dt
        self.z += self.vz*self.dt
        endpoint = np.array([self.x,self.z])
        target = np.array([self.target_x,self.target_z])
        #print("robot v",self.vx,",",self.vz)
        if (np.linalg.norm(target-endpoint) < self.SNAP_THRESH and self.hand_is_closed) or self.has_target:
#        if np.linalg.norm(target-endpoint) < self.SNAP_THRESH:
            self.target_x = self.x
            self.target_z = self.z
            self.has_target = True
        else:
            self.has_target = False

#        if self.z < 0:
#            self.z = 0
#        if self.z > self.AREA_H:
#            self.z = self.AREA_H
#        if self.x > 1.0:
#            self.x = 1.0
#        if self.x < 0.6:
#            self.x = 0.6

        if self.target_z < 0:
            self.target_z = 0

        if self.enable_visualize:
            self.visualize()

    def world_to_img_coord(self,x,z):
        imgx = int(0.5*self.IMAGE_W/self.AREA_W*x+self.IMAGE_W/2)
        imgz = int(self.IMAGE_H - 1.0*self.IMAGE_H/self.AREA_H*z)
        return imgx,imgz

    def get_img(self):
        radius = 2
        canvas = np.zeros((self.IMAGE_H,self.IMAGE_W,3),dtype=np.uint8)

        img_target_x,img_target_z = self.world_to_img_coord(self.target_x,self.target_z)
        cv2.circle(canvas,(img_target_x,img_target_z),radius*2,(0,255,0),-1)

        img_endpoint_x,img_endpoint_z = self.world_to_img_coord(self.x,self.z)
        color = None
        if self.hand_is_closed == 1:
            color = (0,0,255)
        else:
            color = (255,10,0)

        if self.terminate_episode > 0:
            color = (255,255,255)

        cv2.circle(canvas,(img_endpoint_x,img_endpoint_z),radius,color,-1)

        img_line1x,img_line1y = self.world_to_img_coord(-self.AREA_W,0.3)
        img_line2x,img_line2y = self.world_to_img_coord(self.AREA_W,0.3)
        cv2.line(canvas,(img_line1x,img_line1y),(img_line2x,img_line2y),(255,255,255),2)

        return canvas


    def visualize(self):
        canvas = self.get_img()

        winname = "arm"
        cv2.namedWindow(winname)
        cv2.moveWindow(winname, 100, 100)
        cv2.imshow(winname,canvas)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()

    def set_endpoint_v(self,vx,vy,vz,omega,rollrate,pitchrate):
        self.vx = vx
        self.vz = vz

    def move_endpoint_to(self,px,py,pz):
        self.x = px
        self.z = pz

    def get_endpoint_orientation(self):
        return None

    def get_endpoint_position(self):
        return np.array([self.x,0,self.z])

    def set_hand_close(self,set_to_close=1):
        self.hand_is_closed = set_to_close

    def set_target_location(self,px,py,pz=0.0):
        self.target_x = px
        self.target_z = pz

    def set_target_orientation(self,roll,pitch,yaw):
        return

    def set_goal_location(self,px,py,pz):
        return

    def set_goal_orientation(self,roll,pitch,yaw):
        return

    def get_target_location(self):
        return np.array([self.target_x,0,self.target_z])

    def get_target_orientation(self):
        return None

    def set_visualize(self,visualize):
        self.enable_visualize = visualize

    def has_the_target(self):
        return self.has_target

    def update_holding_target(self):
        return

    def holding_the_target(self):
        return self.has_target

    def set_terminate_episode(self,terminate):
        self.terminate_episode = terminate

    def get_terminate_episode(self):
        return self.terminate

    def reset(self,shuffle = True):
        # revert the robot back to original state and randomly place the target around
        location_range = np.zeros(3)
        if shuffle:
            location_range = self.initial_endpoint_range
        location = np.random.uniform(self.initial_endpoint-location_range,self.initial_endpoint+location_range)
        self.move_endpoint_to(location[0],location[1],location[2])
        self.set_hand_close(0)
        self.set_terminate_episode(-1)
        self.has_target = False

        self.vx = 0
        self.vz = 0

        # step the simulation a bit to help stabilize the robot
        for i in range(5):
            self.step()
