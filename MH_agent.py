#!/usr/bin/env python

# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

""" This module implements an agent that roams around a track following random
waypoints and avoiding other vehicles.
The agent also responds to traffic lights. """

import carla
from carla import ColorConverter as cc
#from agents.navigation.agent import Agent, AgentState
from agent import Agent
from local_planner import LocalPlanner
#from agents.navigation.global_route_planner import GlobalRoutePlanner
#from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO
#from agents.navigation.local_planner import RoadOption

import math
import random
import time
import os

import tensorflow as tf

import numpy as np
from PIL import Image
import os
import sys
import math
import random
import cv2
import time
from network import Network
from prepro import Prepro #for preprocessing

class MH_Agent(Agent):
    """
    BasicAgent implements a basic agent that navigates scenes to reach a given
    target destination. This agent respects traffic lights and other vehicles.
    """

    def __init__(self, vehicle, target_speed=20):
        """

        :param vehicle: actor to apply to local planner logic onto
        """
        super(MH_Agent, self).__init__(vehicle)

        self._proximity_threshold = 10.0  # meters
        #self._state = AgentState.NAVIGATING
        args_lateral_dict = {
            'K_P': 1,
            'K_D': 0,
            'K_I': 0.005,
            'dt': 0.2}
        
        """
        self._local_planner = LocalPlanner(
            self._vehicle, opt_dict={'target_speed': target_speed,
                                     'lateral_control_dict': args_lateral_dict})
        """

        #self._hop_resolution = 1.0
        #self._path_seperation_hop = 2
        #self._path_seperation_threshold = 0.5
        self._target_speed = target_speed
        #self._grp = None
        #self.route_trace = None
        ###mh load model ###
        self.model = Network()
        #switch
        #self._image_size = (280,420,3)
        self._image_size = (40,40,3)
        self.front_image = None ##from front_camera?
        ##crystal - 9/28

        #config_gpu = tf.ConfigProto() 
        #config_gpu.gpu_options.visible_device_list = '0' 
        self.pre = Prepro()

    ##mh-crystal
    def run_step(self) :
        
        """
        Execute one step of navigation.
        :return: control
        """
        try:
            control = carla.VehicleControl()
            predict_angle = self.ap(self.front_image)
            control.steer = predict_angle
            control.throttle = 0.3
            return control
        except Exception as e :
            control = carla.VehicleControl()
            control.throttle = 0.0
            print(e)
            return control




    
    ###mh-make ##ap = angle predict
    def ap(self,img):
        try:
            #switch
            #wonjun algorithm
            """
            img = np.reshape(img.raw_data,(280,420,4))
            img = img[:,:,:3][:] #np.array
            images = []
            images.append(img)
            input = np.array(images)
            """
            #input = self.process_image(img)
            input = self.pre.processing(img)
            predict_angle = self.model.angle_predict(input)
            print("-------------------------------------------------------")
            print("predict_angle:",predict_angle)
            #predict = predict_angle[0][0]
            #print("predict:",predict)
            #predict = predict / 100
            #predict = predict * 100
            return(predict_angle)
        except Exception as e :
            print("MH_ap error:",e)
            print("------------")
            return(0)
        


    def is_in_junction(self):
        hlc = self.get_high_level_command()

        if hlc == 4 or hlc == 5 or hlc == 6:
            return False
        else:
            return True

    def process_image(self,img):
            img = np.reshape(img.raw_data,(420,280,4))
            #img = img[:,:,:1]
            img = cv2.resize((cv2.cvtColor(img, cv2.COLOR_RGB2HSV))[:, :, 1], (420, 280))
            img = cv2.resize(img,(420,280))
            img = np.array(img, dtype=np.float32)
            img = np.reshape(img, (-1, 420,280, 1))
            return img


