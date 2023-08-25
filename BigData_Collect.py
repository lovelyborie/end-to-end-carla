import os,sys,random,time
from sqlite3 import Timestamp
from unicodedata import name
import carla
import numpy as np
import cv2
from datetime import datetime
from collections import defaultdict
import pandas as pd
import sys
#settings
import platform
import time
IM_W,IM_H = (150,150)
time_step = 100000
image_save_path ='mydata'
seq_len = 10
number_env_vehicles = 1
global results_dict,results_cols
results_cols = ["index","timestamp","frame_id","filename","angle","speed"]
results_dict = defaultdict(list)
global i_num,episode
i_num = 91006
client = carla.Client('localhost',2000)
world = client.get_world()
new_settings = world.get_settings()
new_settings.synchronous_mode = True
blueprint_library = world.get_blueprint_library()

class Carla_session:

    def __init__(self):
        self.actors = []
        self.counter = 0
        self.n_seq = len(os.listdir(image_save_path))
        self.collision_flag = False
        self.episode_images = []
        self.track_cleanup =[]
        self.env_actors = []

    def add_actors(self):
        start_point = random.choice(world.get_map().get_spawn_points())
        vehicle_bp = blueprint_library.find('vehicle.audi.etron')
        self.vehicle = world.spawn_actor(vehicle_bp,start_point)
        collision_sensor_bp = blueprint_library.find('sensor.other.collision')
        lane_invasion_sensor_bp = blueprint_library.find('sensor.other.lane_invasion')
        camera_sensor_bp = blueprint_library.find('sensor.camera.rgb')
        camera_sensor_bp.set_attribute('image_size_x',str(IM_W))
        camera_sensor_bp.set_attribute('image_size_y',str(IM_H))
        camera_sensor_bp.set_attribute('fov',str(100))
        self.vehicle.set_autopilot(True)
        sensor_location = carla.Transform(carla.Location(x=6,y=0,z=1))
        self.camera = world.spawn_actor(camera_sensor_bp, sensor_location, attach_to = self.vehicle)
        self.collision_sensor = world.spawn_actor(collision_sensor_bp, sensor_location, attach_to = self.vehicle)
        self.actors.extend([self.vehicle,self.camera])
        self.camera.listen(lambda image: self.add_image(image))
    


    def start_new_seq(self):
        self.collision_flag =False
        self.add_actors()
        if not os.path.exists(os.path.join(image_save_path)):
            os.makedirs(os.path.join(image_save_path))
    

    def add_image(self,image):
        try:
            if len(self.actors) > 1 :
                global i_num
                img = np.reshape(image.raw_data,(IM_H,IM_W,4))
                img = img[:,:,:3][:]
                # results_dict['height'].append(img.shape[0])
                # results_dict['width'].append(img.shape[1])
                now = time.time()
                image_filename = str(i_num)
                steer = self.vehicle.get_control().steer
                throttle = self.vehicle.get_control().throttle
                if steer == None or throttle == None :
                    steer = throttle = 0
                results_dict['timestamp'].append(now)
                results_dict['index'].append(now)
                results_dict['filename'].append(image_filename)
                results_dict['frame_id'].append(image_filename)
                results_dict['angle'].append(steer)
                results_dict['speed'].append(throttle)       
                cv2.imwrite(os.path.join(image_save_path,'{}.jpg'.format(image_filename)),img)
                i_num+=1
        except:
            i_num+=1
        if not os.path.exists(os.path.join(image_save_path)):
            os.makedirs(os.path.join(image_save_path))

    def delete_images(self):
        imagestodelete = self.counter-seq_len
        for i in range(self.counter,imagestodelete):
           os.remove(os.path.join(image_save_path,'{}.jpg'.format(i+1)))

    def end_seq(self):
        try:
            self.destroy_actors()
            self.collision_flag =True
            print("end")
        except:
            pass

    def destroy_actors(self):
        try:
            for actor in self.actors:
                actor.destroy()
            self.actors = []
        except:
            pass
       
    def drive_around(self,episodes):
        try:
            self.start_new_seq()
        except:
            pass

def main():
    episode = 0
    c = Carla_session()
    global i_num
    try:
        while episode<10 :
            c.drive_around(episode)
            print("Episode"+str(episode)+" Start")
            while i_num%1000 != 0 :
                blank=0
            c.end_seq()
            print(i_num)
            episode+=1
            i_num+=1
            print("Episode"+str(episode-1)+" is done")
                
            
    finally:
        print("i:",len(results_dict["index"]))
        print("t:",len(results_dict['timestamp']))
        # print("w:",len(results_dict['width']))
        # print("h:",len(results_dict['height']))
        print("f_i:",len(results_dict['frame_id']))
        print("f_n:",len(results_dict['filename']))
        print("a:",len(results_dict['angle']))
        print("s:",len(results_dict['speed']))
        csv_path = os.path.join(image_save_path,'results9.csv')
        results_df = pd.DataFrame(data=results_dict,columns=results_cols)
        results_df.to_csv(csv_path,header=True)

main()
