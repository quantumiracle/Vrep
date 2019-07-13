import sys
import pygame
import numpy as np
import os
import time
from subprocess import Popen
import valuenet
import brain
import vrep
import vrep_sawyer
import simulator
import dqn
import torch
from itertools import count, product

from pygame.locals import *
pygame.init()

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

robot = vrep_sawyer.VrepSawyer(dt=50e-3)
s = simulator.Simulator(robot,dt=50e-3)

# init the pygame
size = width, height = 360, 360
circle_radius = 10
speed = [2, 2]
black = (50,50,50)
gray = (220,220,220)
blue = (0,0,255)
green = (0,255,0)
red = (255,0,0)

screen = pygame.display.set_mode(size)

pygame.display.set_caption("BETA::00.0.1")

clock = pygame.time.Clock()


def text_objects(text, font):
    textSurface = font.render(text, True, black)
    return textSurface, textSurface.get_rect()

def draw_intro():
    screen.fill(gray)

    text = pygame.font.Font('freesansbold.ttf',32)
    TextSurf, TextRect = text_objects("Robot's Endpoint", text)
    TextRect.center = ((width/2),(height/2))
    screen.blit(TextSurf, TextRect)

pressed_up = 0
pressed_down = 0
pressed_left = 0
pressed_right = 0
pressed_fwd = 0
pressed_bwd = 0
pressed_hand = 0
pressed_yawl = 0
pressed_yawr = 0
pressed_rolll = 0
pressed_rollr = 0
pressed_pitchl = 0
pressed_pitchr = 0
endpoint_speed = [0,0,0]
SPEED = 0.3
YAWSPEED = 1.5
ROLLSPEED = 1.5
PITCHSPEED = 1.5

draw_intro()

while True:
    for event in pygame.event.get():
        #print(event)
        if event.type == pygame.QUIT:
             pygame.quit()
             quit()

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                print("left pressed")
                pressed_left = 1
            if event.key == pygame.K_RIGHT:
                print("right pressed")
                pressed_right = 1
            if event.key == pygame.K_UP:
                print("fwd pressed")
                pressed_fwd = 1
            if event.key == pygame.K_DOWN:
                print("bwd pressed")
                pressed_bwd = 1
            if event.key == pygame.K_KP8:
                print("up pressed")
                pressed_up = 1
            if event.key == pygame.K_KP2:
                print("down pressed")
                pressed_down = 1
            if event.key == pygame.K_SPACE:
                print("hand pressed")
                pressed_hand = 1
            if event.key == pygame.K_KP7:
                print("yawl pressed")
                pressed_yawl = 1
            if event.key == pygame.K_KP9:
                print("yawr pressed")
                pressed_yawr = 1
            if event.key == pygame.K_a:
                print("rolll pressed")
                pressed_rolll = 1
            if event.key == pygame.K_d:
                print("rollr pressed")
                pressed_rollr = 1
            if event.key == pygame.K_w:
                print("pitchl pressed")
                pressed_pitchl = 1
            if event.key == pygame.K_s:
                print("pitchr pressed")
                pressed_pitchr = 1
            if event.key == pygame.K_q:
                print("resetting simulation")
                s.reset()

        if event.type == pygame.KEYUP:
            if event.key == pygame.K_LEFT:
                print("left unpressed")
                pressed_left = 0
            if event.key == pygame.K_RIGHT:
                print("right unpressed")
                pressed_right = 0
            if event.key == pygame.K_UP:
                print("fwd unpressed")
                pressed_fwd = 0
            if event.key == pygame.K_DOWN:
                print("bwd unpressed")
                pressed_bwd = 0
            if event.key == pygame.K_KP8:
                print("up unpressed")
                pressed_up = 0
            if event.key == pygame.K_KP2:
                print("down unpressed")
                pressed_down = 0
            if event.key == pygame.K_SPACE:
                print("hand unpressed")
                pressed_hand = 0
            if event.key == pygame.K_KP7:
                print("yawl unpressed")
                pressed_yawl = 0
            if event.key == pygame.K_KP9:
                print("yawr unpressed")
                pressed_yawr = 0
            if event.key == pygame.K_a:
                print("rolll unpressed")
                pressed_rolll = 0
            if event.key == pygame.K_d:
                print("rollr unpressed")
                pressed_rollr = 0
            if event.key == pygame.K_w:
                print("pitchl unpressed")
                pressed_pitchl = 0
            if event.key == pygame.K_s:
                print("pitchr unpressed")
                pressed_pitchr = 0


    draw_intro()
    if(pressed_up == 1):
        pygame.draw.circle(screen, blue,(width/2,height/2 - circle_radius*4 - 3),circle_radius)
    if(pressed_down == 1):
        pygame.draw.circle(screen, blue,(width/2,height/2 + circle_radius*4 + 3),circle_radius)
    if(pressed_left == 1):
        pygame.draw.circle(screen, green,(circle_radius*2 + 3,height/2),circle_radius)
    if(pressed_right == 1):
        pygame.draw.circle(screen, green,(width - circle_radius*2 - 3,height/2),circle_radius)
    if(pressed_fwd == 1):
        pygame.draw.circle(screen, green,(width/2,circle_radius*2+3),circle_radius)
    if(pressed_bwd == 1):
        pygame.draw.circle(screen, green,(width/2,height - circle_radius*2 - 3),circle_radius)
    if(pressed_hand == 1):
        pygame.draw.circle(screen, red,(width/2,height/2),circle_radius)
    if(pressed_yawl == 1):
        pygame.draw.circle(screen, black,(width/2 - circle_radius*2 - 10,height/2),circle_radius)
    if(pressed_yawr == 1):
        pygame.draw.circle(screen, black,(width/2 + circle_radius*2 + 10,height/2),circle_radius)
    if(pressed_rolll == 1):
        pygame.draw.circle(screen, black,(width/2 - circle_radius*2 - 20,height/2 + 10),circle_radius)
    if(pressed_rollr == 1):
        pygame.draw.circle(screen, black,(width/2 + circle_radius*2 + 20,height/2 + 10),circle_radius)
    if(pressed_pitchl == 1):
        pygame.draw.circle(screen, black,(width/2 ,height/2 - circle_radius*2 + 10),circle_radius)
    if(pressed_pitchr == 1):
        pygame.draw.circle(screen, black,(width/2 ,height/2 + circle_radius*2 + 10),circle_radius)

    # set the x,y,z components of the control
    endpoint_speed[0] = (pressed_fwd-pressed_bwd)*SPEED
    endpoint_speed[1] = (pressed_left-pressed_right)*SPEED
    endpoint_speed[2] = (pressed_up-pressed_down)*SPEED
    yaw_speed = (pressed_yawl-pressed_yawr)*YAWSPEED
    roll_speed = (pressed_rolll-pressed_rollr)*ROLLSPEED
    pitch_speed = (pressed_pitchl-pressed_pitchr)*PITCHSPEED

    hand_close = 1 if pressed_hand else 0
#    s.set_control([endpoint_speed[0],endpoint_speed[1],endpoint_speed[2],yaw_speed,roll_speed,pitch_speed,hand_close,1-hand_close])
    s.set_control([endpoint_speed[0],endpoint_speed[1],endpoint_speed[2],-1,hand_close,1-hand_close])
    s.step()

    img = s.r_.get_img()
    depth = s.r_.get_depth()

    if(robot.has_the_target()):
        pygame.draw.circle(screen, blue,(0,0),circle_radius*2)

    pygame.display.update()
