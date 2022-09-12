import os,sys,random,time
import carla
import numpy as np
import cv2

#settings
IM_W,IM_H = (420,280)
time_step = 1.5
image_save_path ='mydata'
seq_len = 10
number_env_vehicles = 1


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
        self.camera.listen(lambda image: self.add_image(image))


    def start_new_seq(self):
        
        self.add_actors()
        self.collision_flag = False
        print('starting new seq')
        #self.counter = 0
        self.n_seq+=1
        print('sequence num:',self.n_seq)
        self.track_cleanup.append(self.n_seq)
        #if not os.path.exists(os.path.join(image_save_path,str(self.n_seq))):
            #os.makedirs(os.path.join(image_save_path,str(self.n_seq)))
        if not os.path.exists(os.path.join(image_save_path)):
            os.makedirs(os.path.join(image_save_path))

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
    
    def delete_images(self):
        imagestodelete = self.counter-seq_len
        for i in range(self.counter,imagestodelete):
            #os.remove(os.path.join(image_save_path,str(self.n_seq),'{}.jpg'.format(i+1)))
           os.remove(os.path.join(image_save_path,'{}.jpg'.format(i+1)))

    def save_images(self):
        for ind,img in enumerate(self.episode_images[-seq_len:]):
            #cv2.imwrite(os.path.join(image_save_path,str(self.n_seq),'{}.jpg'.format(ind)),img)
            cv2.imwrite(os.path.join(image_save_path,str(self.n_seq),'{}.jpg'.format(ind)),img)
    
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
        #self.add_vehicles()
        try:
            self.start_new_seq()
        except:
            pass

        time.sleep(30)


c = Carla_session()
while(True):
    c.drive_around(1)
    c.end_seq()
