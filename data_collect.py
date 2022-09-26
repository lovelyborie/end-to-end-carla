import os,sys,random,time
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
IM_W,IM_H = (420,280)
time_step = 100000
image_save_path ='mydata'
seq_len = 10
number_env_vehicles = 1
global results_dict,results_cols
results_cols = ["index","timestamp","width","height","frame_id","filename","angle","speed"]
results_dict = defaultdict(list)

client = carla.Client('localhost',2000)
#client.set_timeout(5)
world = client.get_world()

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

    def add_vehicles(self):
        env_vehicles_bp = blueprint_library.filter('vehicle.tesla.model3')
        env_vehicles_bp = [x for x in env_vehicles_bp if int(x.get_attribute('number_of_wheels')) == 4]
        env_vehicles_bp = [x for x in env_vehicles_bp if not x.id.endswith('isetta')]
        env_vehicles_bp = [x for x in env_vehicles_bp if not x.id.endswith('carlacola')] 
        spawn_points = world.get_map().get_spawn_points()      
        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor
        self.env_actors = []
        for n, transform in enumerate(spawn_points):
            if n >= number_env_vehicles:
                print("no!")
                break
            env_vehicle_bp = random.choice(env_vehicles_bp)
            if env_vehicle_bp.has_attribute('color'):
                env_vehicle_bp.set_attribute('color', random.choice(env_vehicle_bp.get_attribute('color').recommended_values))
            env_vehicle_bp.set_attribute('role_name', 'autopilot')
            env_vehicle = world.spawn_actor(env_vehicle_bp,transform)
            env_vehicle.set_autopilot(True)
            self.env_actors.append(env_vehicle)

    def add_actors(self):

        start_point = random.choice(world.get_map().get_spawn_points())
        start_point.location.x=4.767345
        start_point.location.y=-43.606983
        start_point.location.z=0.281942
        start_point.rotation.pitch = 0.000000
        start_point.rotation.yaw = -90.224854
        start_point.rotation.roll = 0.000000

        vehicle_bp = blueprint_library.find('vehicle.audi.etron')
        self.vehicle = world.spawn_actor(vehicle_bp,start_point)
        collision_sensor_bp = blueprint_library.find('sensor.other.collision')
        lane_invasion_sensor_bp = blueprint_library.find('sensor.other.lane_invasion')
        camera_sensor_bp = blueprint_library.find('sensor.camera.rgb')
        camera_sensor_bp.set_attribute('image_size_x',str(IM_W))
        camera_sensor_bp.set_attribute('image_size_y',str(IM_H))
        camera_sensor_bp.set_attribute('fov',str(100))
        self.vehicle.set_autopilot(True)
        #sensor_location = carla.Transform(carla.Location(x=4,y=0,z=2.5))
        sensor_location = carla.Transform(carla.Location(x=6,y=0,z=1))
        self.camera = world.spawn_actor(camera_sensor_bp, sensor_location, attach_to = self.vehicle)
        self.collision_sensor = world.spawn_actor(collision_sensor_bp, sensor_location, attach_to = self.vehicle)
        self.actors.extend([self.vehicle,self.camera])
        ##mh##
        self.camera.listen(lambda image: self.add_image(image))
    


    def start_new_seq(self):
        
        self.add_actors()
        #self.collision_flag = False
        #print('starting new seq')
        #self.counter = 0
        #self.n_seq+=1
        #print('sequence num:',self.n_seq)
        #self.track_cleanup.append(self.n_seq)
        #if not os.path.exists(os.path.join(image_save_path,str(self.n_seq))):
            #os.makedirs(os.path.join(image_save_path,str(self.n_seq)))
        if not os.path.exists(os.path.join(image_save_path)):
            os.makedirs(os.path.join(image_save_path))
    
    """ mh-before
    def add_image(self,image):
        self.counter += 1
        img = np.reshape(image.raw_data,(IM_H,IM_W,4))
        img = img[:,:,:3][:]
        
        #cv2.imwrite(os.path.join(image_save_path,str(self.n_seq),'{}.jpg'.format(self.counter)),img)
        cv2.imwrite(os.path.join(image_save_path,'{}.jpg'.format(self.counter)),img)
        #if self.counter%1 == 0:
            #self.n_seq += 1
            #self.counter = 0
            #if not os.path.exists(os.path.join(image_save_path,str(self.n_seq))):
                #os.makedirs(os.path.join(image_save_path,str(self.n_seq)))
        if not os.path.exists(os.path.join(image_save_path)):
            os.makedirs(os.path.join(image_save_path))
    """
    
    #mh-for_csv

    def add_image(self,image):
        img = np.reshape(image.raw_data,(IM_H,IM_W,4))
        img = img[:,:,:3][:]
        results_dict['height'].append(img.shape[0])
        results_dict['width'].append(img.shape[1])
        now = time.time()
        image_filename = str(now)
        #results_dict['timestamp'].append(time.time())
        results_dict['timestamp'].append(now)
        results_dict['index'].append(now)
        results_dict['filename'].append(image_filename)
        results_dict['frame_id'].append(now)
        results_dict['angle'].append(self.vehicle.get_control().steer)
        results_dict['speed'].append(self.vehicle.get_control().throttle)       
        #Checking stter,throttle in carla vehicle ##mh
        #d = self.vehicle.get_control().steer
        #e = self.vehicle.get_control().throttle

        cv2.imwrite(os.path.join(image_save_path,'{}.jpg'.format(now)),img)
        if not os.path.exists(os.path.join(image_save_path)):
            os.makedirs(os.path.join(image_save_path))

    def delete_images(self):
        imagestodelete = self.counter-seq_len
        for i in range(self.counter,imagestodelete):
            #os.remove(os.path.join(image_save_path,str(self.n_seq),'{}.jpg'.format(i+1)))
           os.remove(os.path.join(image_save_path,'{}.jpg'.format(i+1)))

    """
    def save_images(self):
        for ind,img in enumerate(self.episode_images[-seq_len:]):
            #cv2.imwrite(os.path.join(image_save_path,str(self.n_seq),'{}.jpg'.format(ind)),img)
            cv2.imwrite(os.path.join(image_save_path,str(self.n_seq),'{}.jpg'.format(ind)),img)
    """   
    
    def end_seq(self):
        self.destroy_actors()
        self.collision_flag =True
        print("end")
        #self.delete_images()

    def destroy_actors(self):
        for actor in self.actors:
            actor.destroy()

        self.actors = []
    
    def get_directions(self):
        thr = random.choice([0.8,0.7,0.6])
        steer = random.choice([-0.3,0.0,0.0,0.0,0.3,0.1,-0.1])
        return carla.VehicleControl(thr,steer)  
       
    def drive_around(self,episodes):
        try:
            self.start_new_seq()
        except:
            pass

        time.sleep(1000)

"""
c = Carla_session()
while(True):
    c.drive_around(1)
    c.end_seq()
"""

def main():
    c = Carla_session()
    #for i in range(100):
    try:
        #c.start_new_seq()
        c.drive_around(1000)
        for i in range(10000):
            time.sleep(10)

    #print(platform.python_version())
    finally:
        csv_path = os.path.join(image_save_path,'results.csv')
        results_df = pd.DataFrame(data=results_dict,columns=results_cols)
        results_df.to_csv(csv_path,header=True)

main()