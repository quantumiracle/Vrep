import cv2
import numpy as np

class Simulator:
    ''' 
    the environment class with python interface,
    interaction with vrep through the robot_arm
    ''' 

    def __init__(self,robot_arm,dt,target_x=0,target_y=0,target_z=0,visualize=True,goal_x=0,goal_y=0,goal_z=0):
        # for now, let the control be just the velocity of the endpoint
        self.r_ = robot_arm

        self.r_.dt = dt

        self.DONE_DISTANCE = 0.25
        self.MAX_SPEED = 1.2
        self.MAX_OMEGA = 0.6

        # now we will have to define the arm reach ourselves, for now, let's use 1
        self.r_.set_target_location(target_x,target_y,target_z)
        self.set_goal(goal_x,goal_y,goal_z)

        # create gray background
        self.set_visualize(visualize)

        self.is_discrete = False
        self.DISCRETE_A = 3
        self.lifting_up = False
        self.will_grasp = True # if set to false the hand will just go to the object and lift up without closing the hand

        self.TRAY_W = self.r_.TRAY_DY
        self.TRAY_H = self.r_.TRAY_DX
        self.TRAY_X = self.r_.TRAY_X
        self.TRAY_Y = self.r_.TRAY_Y

    def get_optimal_control(self,error=0.0):
        # now let the action be [v(3),yaw,gripper(open,close)]

        # call vrep to just move to the point and grab the object
        current_endpoint = self.r_.get_endpoint_position()
        current_orientation = self.r_.get_endpoint_orientation()
        target_vector = self.r_.get_target_location()
        target_orientation = self.r_.get_target_orientation()

        HAND_HOVER_DISTANCE = 0.07
        GRASP_THRESH_DIST = 0.01
        GRASP_THRESH_ORIENTATION = 0.05
        ORIENTATION_OFFSET_VECTOR = np.array([0,-np.pi/2.0,0])
        GRASP_DURATION = 15
        NOISE_RATIO = 0.05

        grasp = np.array([0])
        difference_vector_v = target_vector - current_endpoint + np.array([error,0,HAND_HOVER_DISTANCE])
#        orientation_vector_w = target_orientation + ORIENTATION_OFFSET_VECTOR - current_orientation

        if(np.linalg.norm(difference_vector_v[0:2])> GRASP_THRESH_DIST and (not self.r_.holding_the_target())):
            # set the endpoint velocity to 1
#            print("siml: stage1, moving to hover position")
            noise_mean = np.zeros(3)
            noise_cov = np.eye(3)
            noise_cov[2,2] = 0
            difference_vector_v = np.array([difference_vector_v[0],difference_vector_v[1],0]) # for moving arm to the hover position
            orientation_vector_w = np.array([0,0,0])
            grasp = [0,1]
#            print("siml: stage1, current distance",np.linalg.norm(difference_vector_v[0:2]))
#            print("siml: stage1, reach",self.MAX_SPEED*self.r_.dt)
            if self.MAX_SPEED*self.r_.dt < np.linalg.norm(difference_vector_v[0:2]):
                #too far, go with max speed
                difference_vector_v = difference_vector_v / np.linalg.norm(difference_vector_v)*self.MAX_SPEED + np.random.multivariate_normal(noise_mean,noise_cov)*self.MAX_SPEED
                # print("siml: stage1 out of reach",difference_vector_v)
            else:
                # target within reach, go with speed just enough to reach target
                difference_vector_v = difference_vector_v / self.r_.dt
#                print("siml: stage1 within reach",difference_vector_v)

        # if the robot is at the target, the grasp it not self.r_.has_the_target() and
        # hard coded for now, will explore the use of force sensors on the fingers later
        elif(not self.lifting_up):
#            print("siml, stage2, grasping",self.r_.holding_the_target(),"",self.lifting_up)
            difference_vector_v = np.array([0,0,1])*(difference_vector_v[2]-np.array([0,0,0.05]))
            orientation_vector_w = np.array([0,0,0])
            grasp = [0,1]
#            print("siml: stage2",abs((target_vector - current_endpoint)[2]))
            if self.MAX_SPEED*self.r_.dt < np.linalg.norm(difference_vector_v[2]):
                #too far, go with max speed
                difference_vector_v = difference_vector_v / np.linalg.norm(difference_vector_v)*self.MAX_SPEED + np.random.normal(0.0,NOISE_RATIO)*self.MAX_SPEED
#                print("siml: stage2 out of reach",difference_vector_v)
            else:
                # target within reach, go with speed just enough to reach target
                difference_vector_v = difference_vector_v / self.r_.dt
#                print("siml: stage2 within reach",difference_vector_v)

            if(abs((target_vector - current_endpoint)[2]) < HAND_HOVER_DISTANCE and not self.lifting_up):
                difference_vector_v = np.zeros(3)
                if self.will_grasp == 1:
                    grasp = [1,0]
    #                print("siml: stage2 closing hand")
                    if self.r_.holding_the_target():
                        self.lifting_up = True
    #                    print("siml: stage2 ready to lift up")
                else:
                    grasp = [0,1]
                    self.lifting_up = True


        # if the robot is already holding the target
        else:
#            print("siml, stage3, lifting")
            difference_vector_v = np.array([0,0,1])*(self.DONE_DISTANCE+0.08 - current_endpoint[2])
            orientation_vector_w = np.array([0,0,0])
            if self.will_grasp == 1:
                grasp = [1,0]
            else:
                grasp = [0,1]
            if self.MAX_SPEED*self.r_.dt < np.linalg.norm(difference_vector_v[2]):
                #too far, go with max speed
                difference_vector_v = difference_vector_v / np.linalg.norm(difference_vector_v)*self.MAX_SPEED + np.random.normal(0.0,NOISE_RATIO)*self.MAX_SPEED
#                print("siml: stage3 out of reach",difference_vector_v)
            else:
                # target within reach, go with speed just enough to reach target
                difference_vector_v = difference_vector_v / self.r_.dt
#                print("siml: stage3 within reach",difference_vector_v)

#        print("siml: stage0",current_endpoint[2],",",self.lifting_up,",",self.r_.holding_the_target())
        terminate = -1
        if(current_endpoint[2] > self.DONE_DISTANCE + 0.05 and self.lifting_up and self.r_.holding_the_target()):
            terminate = 1

#        print("====================================================")
#        print("v",difference_vector_v)
        #return difference_vector_v[[0,2]]
#        print(np.concatenate((difference_vector_v,[terminate],grasp),axis=0))
        return np.concatenate((difference_vector_v,[terminate],grasp),axis=0)
#        return np.concatenate((difference_vector_v,orientation_vector_w,grasp),axis=0)

    def set_goal(self,goal_x,goal_y,goal_z):
        self.goal_x = goal_x
        self.goal_y = goal_y
        self.goal_z = goal_z
        goal_vector = np.array([self.goal_x,self.goal_y,self.goal_z])
        if(np.linalg.norm(goal_vector) > self.r_.arm_reach):
            print("simulator: warning, goal is out of reach ")

    def set_visualize(self,visualize=True):
        self.r_.set_visualize(visualize)

    def get_robot_state(self):
        # use the numerical states for now
        end_effector = self.r_.get_endpoint_position()
        end_effector_orientation = self.r_.get_endpoint_orientation()
        target = self.r_.get_target_location()
        target_orientation = self.r_.get_target_orientation()
        distance = target-end_effector
        grasp = self.r_.hand_is_closed
        has_target = self.r_.holding_the_target()

        img_state = self.r_.get_img()

        numerical_state = np.array([
            end_effector[0],
            end_effector[1],
            end_effector[2],
#            end_effector_orientation[0],
#            end_effector_orientation[1],
#            end_effector_orientation[2],
            distance[0],# target[0],
            distance[1],# target[1],
            distance[2],# target[2],
            target[2],
#            target_orientation[0],
#            target_orientation[1],
#            target_orientation[2],
            has_target,
            grasp # the gripper state is reported as 0 or 1
        ])

#        print("Simulator: state",state)
#        return img_state, numerical_state
        return img_state, numerical_state

#     def get_reward_and_done(self,state):
#         MAX_REWARD = 3.0
#         MIN_REWARD = -1.0*MAX_REWARD

#         target = self.r_.get_target_location()
#         current = self.r_.get_endpoint_position()

#         current_distance = np.linalg.norm(target-current)
#         # print('current distance: ', current_distance)
#         done = False

#         reward = -0.05

#         OB_MARGIN = 0.05
#         is_ob = False
# #        # if the hand is out of the box, terminate
#         if(current[0] < self.TRAY_X-self.TRAY_H/2 - OB_MARGIN):
# #            print("siml: ob: x--")
#             is_ob = True
#         elif current[0] > self.TRAY_X+self.TRAY_H/2 + OB_MARGIN:
# #            print("siml: ob: x++")
#             is_ob = True
#         elif current[1] < self.TRAY_Y-self.TRAY_W/2 - OB_MARGIN:
# #            print("siml: ob: y--")
#             is_ob = True
#         elif current[1] > self.TRAY_Y+self.TRAY_W/2 + OB_MARGIN:
# #            print("siml: ob: y++")
#             is_ob = True
#         elif current[2] > self.DONE_DISTANCE + 0.3:
# #            print("siml: ob: z++")
#             is_ob = True
#         elif current[2] < 0.0:
# #            print("siml: ob: z--")
#             is_ob = True

#         if is_ob:
#             done = True
#             reward = -2.5
# #            print("simulator: reward:",reward)
#             return reward,done

#         # all dense reward are normalized by the done distance, the reason there is -1 in the first case is to encourage grasping to get the target, the agent will get +1 reward as soo as it closes hand at the target
#         reward = -1*current_distance/self.DONE_DISTANCE + -1*(self.DONE_DISTANCE - target[2])/self.DONE_DISTANCE

#         reward *= 0.1 #scaling to help learning

#         if self.r_.holding_the_target():
#             reward *= 0.9 #if we are holding the target, we get an immediate 10% penalty reduction

#         if self.r_.terminate_episode > 0:
#             done = True
#             self.lifting_up = False
#             if(target[2] > self.DONE_DISTANCE and self.r_.holding_the_target()):
# #                print("simulator: grasping and lifting done, rewarded 1")
#                 reward = 1.0
#             else:
# #                print("simulator: wrong termination, rewarded -1",target[2],",",self.r_.holding_the_target())
#                 reward = -2.5
# #            print("simulator: reward:",reward)
#             return reward,done

# #        print("simulator: reward:",reward)
#         return reward,done


    def get_reward_and_done(self,state):
        target = self.r_.get_target_location()
        current = self.r_.get_endpoint_position()
        current_distance = np.linalg.norm(target-current)

        reward = 0.
        # OB_MARGIN = 0.05
        # is_ob = False
        done = False
# #        # if the hand is out of the box, terminate
#         if(current[0] < self.TRAY_X-self.TRAY_H/2 - OB_MARGIN):
# #            print("siml: ob: x--")
#             is_ob = True
#         elif current[0] > self.TRAY_X+self.TRAY_H/2 + OB_MARGIN:
# #            print("siml: ob: x++")
#             is_ob = True
#         elif current[1] < self.TRAY_Y-self.TRAY_W/2 - OB_MARGIN:
# #            print("siml: ob: y--")
#             is_ob = True
#         elif current[1] > self.TRAY_Y+self.TRAY_W/2 + OB_MARGIN:
# #            print("siml: ob: y++")
#             is_ob = True
#         elif current[2] > self.DONE_DISTANCE + 0.3:
# #            print("siml: ob: z++")
#             is_ob = True
#         elif current[2] < 0.0:
# #            print("siml: ob: z--")
#             is_ob = True

#         if is_ob:
#             done = True
#             reward -= -2.5

        distance_punishment = 0.1/(current_distance+0.1)+ 0.1/(np.abs(self.DONE_DISTANCE - target[2])+0.1)
        # print('distance_punishment', distance_punishment)
        reward += distance_punishment
        if self.r_.holding_the_target():
            reward += 2.0

        if(target[2] > self.DONE_DISTANCE and self.r_.holding_the_target()):
            reward += 50.0
            print('Get The Goal!')
            
        if self.r_.terminate_episode > 0:
            done = True
#             self.lifting_up = False
#             if(target[2] > self.DONE_DISTANCE and self.r_.holding_the_target()):
#                 reward += 10.0
#             else:
# #                print("simulator: wrong termination, rewarded -1",target[2],",",self.r_.holding_the_target())
#                 reward -=10
        return reward, done



    def set_control(self,controls):
        # [speed3, omega3, ..., terminate, hand_close, hand_open]
        # for now we'll not control the yaw
        # the speed is defined as the percentage of the max speed
        velocity = np.array(controls[0:3])
        if np.linalg.norm(velocity) > self.MAX_SPEED:
            velocity = velocity / np.linalg.norm(velocity) * self.MAX_SPEED

        self.r_.set_endpoint_v(
            velocity[0],
            velocity[1],
            velocity[2],
            0,
            0,
            0)
#            controls[3]*self.MAX_OMEGA,
#            controls[4]*self.MAX_OMEGA,
#            controls[5]*self.MAX_OMEGA)
        # define the last two commands as close / open the gripper
        if(controls[-2] > controls[-1]):
            self.r_.set_hand_close(set_to_close=1)
        else:
            self.r_.set_hand_close(set_to_close=0)

        # print('control: ', controls[-3])
        if(controls[-3] <= 0):
            # terminate the episode
            self.r_.set_terminate_episode(1)
        else:
            self.r_.set_terminate_episode(-1)

    def step(self):
        self.r_.step()

    def reset(self):
        self.will_grasp = np.random.randint(2)
        self.lifting_up = False
        self.randomly_place_target()
        self.randomly_place_goal()
        self.r_.reset()

    def randomly_place_target(self):
        #randomly place the target in the tray of reach
#        x_target = np.random.uniform(self.TRAY_X-self.TRAY_H/2,self.TRAY_X+self.TRAY_H/2)
#        y_target = np.random.uniform(self.TRAY_Y-self.TRAY_W/2,self.TRAY_Y+self.TRAY_W/2)
        x_target = np.random.uniform(self.TRAY_X-0.03,self.TRAY_X+0.03)
        y_target = np.random.uniform(self.TRAY_Y-0.03,self.TRAY_Y+0.03)

        z_target = 0.05

        # random the rotation
        roll_target = 0#np.random.uniform(-np.pi,np.pi)
        pitch_target = 0#np.random.uniform(-np.pi,np.pi)
        yaw_target = 0#np.random.uniform(-np.pi,np.pi)

        # for now lets just use the height of the target to 0 (the ground level)
        self.r_.set_target_location(x_target,y_target,z_target)
        self.r_.set_target_orientation(roll_target,pitch_target,yaw_target)

        return x_target,y_target,z_target

    def randomly_place_goal(self):
        # randomly place the target in the tray of reach
        x_target = np.random.uniform(self.TRAY_X-self.TRAY_H/2,self.TRAY_X+self.TRAY_H/2)
        y_target = np.random.uniform(self.TRAY_Y-self.TRAY_W/2,self.TRAY_Y+self.TRAY_W/2)
        z_target = 0.0

        self.set_goal(x_target,y_target,z_target) # this method is probably outdated
        self.r_.set_goal_location(x_target,y_target,z_target)

        return x_target,y_target,z_target
