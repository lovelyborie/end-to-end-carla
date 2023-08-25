#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Allows controlling a vehicle with a keyboard. For a simpler and more
# documented example, please take a look at tutorial.py.

"""
Welcome to CARLA manual control.

Use ARROWS or WASD keys for control.

    W            : throttle
    S            : brake
    A/D          : steer left/right
    Q            : toggle reverse
    Space        : hand-brake
    P            : toggle autopilot
    M            : toggle manual transmission
    ,/.          : gear up/down
    CTRL + W     : toggle constant velocity mode at 60 km/h

    L            : toggle next light type
    SHIFT + L    : toggle high beam
    Z/X          : toggle right/left blinker
    I            : toggle interior light

    TAB          : change sensor position
    ` or N       : next sensor
    [1-9]        : change to sensor [1-9]
    G            : toggle radar visualization
    C            : change weather (Shift+C reverse)
    Backspace    : change vehicle

    O            : open/close all doors of vehicle
    T            : toggle vehicle's telemetry

    V            : Select next map layer (Shift+V reverse)
    B            : Load current selected map layer (Shift+B to unload)

    R            : toggle recording images to disk

    CTRL + R     : toggle recording of simulation (replacing any previous)
    CTRL + P     : start replaying last recorded simulation
    CTRL + +     : increments the start time of the replay by 1 second (+SHIFT = 10 seconds)
    CTRL + -     : decrements the start time of the replay by 1 second (+SHIFT = 10 seconds)

    F1           : toggle HUD
    H/?          : toggle help
    ESC          : quit
"""

from __future__ import print_function



# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================


import glob
import os
import sys
import time 
#from carla_vision_keeping import calculate_velocity_and_angle

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================


import carla


from carla import ColorConverter as cc

import argparse
import collections
import datetime
import logging
import math
import random
import re
import weakref
#minhong 
import cv2
from enum import Enum
import warnings
warnings.filterwarnings("ignore")
try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import KMOD_SHIFT
    from pygame.locals import K_0
    from pygame.locals import K_9
    from pygame.locals import K_BACKQUOTE
    from pygame.locals import K_BACKSPACE
    from pygame.locals import K_COMMA
    from pygame.locals import K_DOWN
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_F1
    from pygame.locals import K_LEFT
    from pygame.locals import K_PERIOD
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SLASH
    from pygame.locals import K_SPACE
    from pygame.locals import K_TAB
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_b
    from pygame.locals import K_c
    from pygame.locals import K_d
    from pygame.locals import K_g
    from pygame.locals import K_h
    from pygame.locals import K_i
    from pygame.locals import K_l
    from pygame.locals import K_m
    from pygame.locals import K_n
    from pygame.locals import K_o
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_r
    from pygame.locals import K_s
    from pygame.locals import K_t
    from pygame.locals import K_v
    from pygame.locals import K_w
    from pygame.locals import K_x
    from pygame.locals import K_z
    from pygame.locals import K_MINUS
    from pygame.locals import K_EQUALS
    from pygame.locals import K_e

except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

# ==============================================================================
# -- minhong camera
my_img = np.zeros(shape=(800,600,3),dtype=np.uint8)
my_img = None
# ==============================================================================

# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================


def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name

def get_actor_blueprints(world, filter, generation):
    #bps = world.get_blueprint_library().filter(filter)
    bps = world.get_blueprint_library().filter('vehicle.bmw.*')

    if generation.lower() == "all":
        return bps

    # If the filter returns only one bp, we assume that this one needed
    # and therefore, we ignore the generation
    if len(bps) == 1:
        return bps

    try:
        int_generation = int(generation)
        # Check if generation is in available generations
        if int_generation in [1, 2]:
            bps = [x for x in bps if int(x.get_attribute('generation')) == int_generation]
            return bps
        else:
            print("   Warning! Actor Generation is not valid. No actor will be spawned.")
            return []
    except:
        print("   Warning! Actor Generation is not valid. No actor will be spawned.")
        return []


# ==============================================================================
# -- World ---------------------------------------------------------------------
# ==============================================================================


class World(object):
    def __init__(self, carla_world, hud, args):
        self.world = carla_world
        self.sync = args.sync
        self.actor_role_name = args.rolename
        try:
            self.map = self.world.get_map()
        except RuntimeError as error:
            print('RuntimeError: {}'.format(error))
            print('  The server could not send the OpenDRIVE (.xodr) file:')
            print('  Make sure it exists, has the same name of your town, and is correct.')
            sys.exit(1)
        self.hud = hud
        self.player = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.imu_sensor = None
        self.radar_sensor = None
        self.camera_manager = None
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self._actor_filter = args.filter
        self._actor_generation = args.generation
        self._gamma = args.gamma
        self.restart()
        self.world.on_tick(hud.on_world_tick)
        self.recording_enabled = False
        self.recording_start = 0
        self.constant_velocity_enabled = False
        self.show_vehicle_telemetry = False
        self.doors_are_open = False
        self.current_map_layer = 0
        #add
        #blueprint_library = self.world.get_blueprint_library()
        self.map_layer_names = [
            carla.MapLayer.NONE,
            carla.MapLayer.Buildings,
            carla.MapLayer.Decals,
            carla.MapLayer.Foliage,
            carla.MapLayer.Ground,
            carla.MapLayer.ParkedVehicles,
            carla.MapLayer.Particles,
            carla.MapLayer.Props,
            carla.MapLayer.StreetLights,
            carla.MapLayer.Walls,
            carla.MapLayer.All
        ]

    def restart(self):
        global my_img
        #blueprint_library = self.world.get_blueprint_library()
        self.player_max_speed = 1.589
        self.player_max_speed_fast = 3.713
        # Keep same camera config if the camera manager exists.
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_index = self.camera_manager.transform_index if self.camera_manager is not None else 0
        # Get a random blueprint.
        blueprint = random.choice(get_actor_blueprints(self.world, self._actor_filter, self._actor_generation))
        #blueprint = random.choice(get_actor_blueprints(self.world, blueprint_library.filter('vehicle.bmw.*'), self._actor_generation))

        blueprint.set_attribute('role_name', self.actor_role_name)
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        if blueprint.has_attribute('driver_id'):
            driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
            blueprint.set_attribute('driver_id', driver_id)
        if blueprint.has_attribute('is_invincible'):
            blueprint.set_attribute('is_invincible', 'true')
        # set the max speed
        if blueprint.has_attribute('speed'):
            self.player_max_speed = float(blueprint.get_attribute('speed').recommended_values[1])
            self.player_max_speed_fast = float(blueprint.get_attribute('speed').recommended_values[2])

        # Spawn the player.
        if self.player is not None:
            spawn_point = self.player.get_transform()
            spawn_point.location.z += 2.0
            spawn_point.rotation.roll = 0.0
            spawn_point.rotation.pitch = 0.0
            self.destroy()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
            self.show_vehicle_telemetry = False
            self.modify_vehicle_physics(self.player)
        while self.player is None:
            if not self.map.get_spawn_points():
                print('There are no spawn points available in your map/town.')
                print('Please add some Vehicle Spawn Point to your UE4 scene.')
                sys.exit(1)
            spawn_points = self.map.get_spawn_points()
            spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
            #for minhong right turn
            spawn_point.location.x=4.767345
            spawn_point.location.y=-43.606983
            spawn_point.location.z=0.281942
            
            #for minhong left turn
            #spawn_point.location.x = 408.9937744140625
            #spawn_point.location.y = -70.99147033691406
            #spawn_point.location.z = 0.2819424271583557


            spawn_point.rotation.pitch = 0.000000
            spawn_point.rotation.yaw = -90.224854
            spawn_point.rotation.roll = 0.000000

            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
            self.show_vehicle_telemetry = False
            self.modify_vehicle_physics(self.player)
        # Set up the sensors.
        self.collision_sensor = CollisionSensor(self.player, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        self.gnss_sensor = GnssSensor(self.player)
        self.imu_sensor = IMUSensor(self.player)
        self.camera_manager = CameraManager(self.player, self.hud, self._gamma)
        self.camera_manager.transform_index = cam_pos_index
        self.camera_manager.set_sensor(cam_index, notify=False)
        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)

        if self.sync:
            self.world.tick()
        else:
            self.world.wait_for_tick()

    def next_weather(self, reverse=False):
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.hud.notification('Weather: %s' % preset[1])
        self.player.get_world().set_weather(preset[0])

    def next_map_layer(self, reverse=False):
        self.current_map_layer += -1 if reverse else 1
        self.current_map_layer %= len(self.map_layer_names)
        selected = self.map_layer_names[self.current_map_layer]
        self.hud.notification('LayerMap selected: %s' % selected)

    def load_map_layer(self, unload=False):
        selected = self.map_layer_names[self.current_map_layer]
        if unload:
            self.hud.notification('Unloading map layer: %s' % selected)
            self.world.unload_map_layer(selected)
        else:
            self.hud.notification('Loading map layer: %s' % selected)
            self.world.load_map_layer(selected)

    def toggle_radar(self):
        if self.radar_sensor is None:
            self.radar_sensor = RadarSensor(self.player)
        elif self.radar_sensor.sensor is not None:
            self.radar_sensor.sensor.destroy()
            self.radar_sensor = None

    def modify_vehicle_physics(self, actor):
        #If actor is not a vehicle, we cannot use the physics control
        try:
            physics_control = actor.get_physics_control()
            physics_control.use_sweep_wheel_collision = True
            actor.apply_physics_control(physics_control)
        except Exception:
            pass

    def tick(self, clock):
        self.hud.tick(self, clock)

    def render(self, display):
        self.camera_manager.render(display)
        self.hud.render(display)

    def destroy_sensors(self):
        self.camera_manager.sensor.destroy()
        self.camera_manager.sensor = None
        self.camera_manager.index = None

    def destroy(self):
        if self.radar_sensor is not None:
            self.toggle_radar()
        sensors = [
            self.camera_manager.sensor,
            self.collision_sensor.sensor,
            self.lane_invasion_sensor.sensor,
            self.gnss_sensor.sensor,
            self.imu_sensor.sensor]
        for sensor in sensors:
            if sensor is not None:
                sensor.stop()
                sensor.destroy()
        if self.player is not None:
            self.player.destroy()


# ==============================================================================
# -- KeyboardControl -----------------------------------------------------------
# ==============================================================================


class KeyboardControl(object):
    """Class that handles keyboard input."""
    def __init__(self, world, start_in_autopilot):
        self._autopilot_enabled = start_in_autopilot
        if isinstance(world.player, carla.Vehicle):
            self._control = carla.VehicleControl()
            self._lights = carla.VehicleLightState.NONE
            world.player.set_autopilot(self._autopilot_enabled)
            world.player.set_light_state(self._lights)
        elif isinstance(world.player, carla.Walker):
            self._control = carla.WalkerControl()
            self._autopilot_enabled = False
            self._rotation = world.player.get_transform().rotation
        else:
            raise NotImplementedError("Actor type not supported")
        self._steer_cache = 0.0
        world.hud.notification("Press 'H' or '?' for help.", seconds=4.0)

    def parse_events(self, client, world, clock, sync_mode):
        if isinstance(self._control, carla.VehicleControl):
            current_lights = self._lights
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True
                elif event.key == K_BACKSPACE:
                    if self._autopilot_enabled:
                        world.player.set_autopilot(False)
                        world.restart()
                        world.player.set_autopilot(True)
                    else:
                        world.restart()
                elif event.key == K_F1:
                    world.hud.toggle_info()
                elif event.key == K_v and pygame.key.get_mods() & KMOD_SHIFT:
                    world.next_map_layer(reverse=True)
                elif event.key == K_v:
                    world.next_map_layer()
                elif event.key == K_b and pygame.key.get_mods() & KMOD_SHIFT:
                    world.load_map_layer(unload=True)
                elif event.key == K_b:
                    world.load_map_layer()
                elif event.key == K_h or (event.key == K_SLASH and pygame.key.get_mods() & KMOD_SHIFT):
                    world.hud.help.toggle()
                elif event.key == K_TAB:
                    world.camera_manager.toggle_camera()
                elif event.key == K_c and pygame.key.get_mods() & KMOD_SHIFT:
                    world.next_weather(reverse=True)
                elif event.key == K_c:
                    world.next_weather()
                elif event.key == K_g:
                    world.toggle_radar()
                elif event.key == K_BACKQUOTE:
                    world.camera_manager.next_sensor()
                elif event.key == K_n:
                    world.camera_manager.next_sensor()
                elif event.key == K_w and (pygame.key.get_mods() & KMOD_CTRL):
                    if world.constant_velocity_enabled:
                        world.player.disable_constant_velocity()
                        world.constant_velocity_enabled = False
                        world.hud.notification("Disabled Constant Velocity Mode")
                    else:
                        world.player.enable_constant_velocity(carla.Vector3D(17, 0, 0))
                        world.constant_velocity_enabled = True
                        world.hud.notification("Enabled Constant Velocity Mode at 60 km/h")
                elif event.key == K_o:
                    try:
                        if world.doors_are_open:
                            world.hud.notification("Closing Doors")
                            world.doors_are_open = False
                            world.player.close_door(carla.VehicleDoor.All)
                        else:
                            world.hud.notification("Opening doors")
                            world.doors_are_open = True
                            world.player.open_door(carla.VehicleDoor.All)
                    except Exception:
                        pass
                elif event.key == K_t:
                    if world.show_vehicle_telemetry:
                        world.player.show_debug_telemetry(False)
                        world.show_vehicle_telemetry = False
                        world.hud.notification("Disabled Vehicle Telemetry")
                    else:
                        try:
                            world.player.show_debug_telemetry(True)
                            world.show_vehicle_telemetry = True
                            world.hud.notification("Enabled Vehicle Telemetry")
                        except Exception:
                            pass
                elif event.key > K_0 and event.key <= K_9:
                    index_ctrl = 0
                    if pygame.key.get_mods() & KMOD_CTRL:
                        index_ctrl = 9
                    world.camera_manager.set_sensor(event.key - 1 - K_0 + index_ctrl)
                elif event.key == K_r and not (pygame.key.get_mods() & KMOD_CTRL):
                    world.camera_manager.toggle_recording()
                elif event.key == K_r and (pygame.key.get_mods() & KMOD_CTRL):
                    if (world.recording_enabled):
                        client.stop_recorder()
                        world.recording_enabled = False
                        world.hud.notification("Recorder is OFF")
                    else:
                        client.start_recorder("manual_recording.rec")
                        world.recording_enabled = True
                        world.hud.notification("Recorder is ON")
                elif event.key == K_p and (pygame.key.get_mods() & KMOD_CTRL):
                    # stop recorder
                    client.stop_recorder()
                    world.recording_enabled = False
                    # work around to fix camera at start of replaying
                    current_index = world.camera_manager.index
                    world.destroy_sensors()
                    # disable autopilot
                    self._autopilot_enabled = False
                    world.player.set_autopilot(self._autopilot_enabled)
                    world.hud.notification("Replaying file 'manual_recording.rec'")
                    # replayer
                    client.replay_file("manual_recording.rec", world.recording_start, 0, 0)
                    world.camera_manager.set_sensor(current_index)
                elif event.key == K_MINUS and (pygame.key.get_mods() & KMOD_CTRL):
                    if pygame.key.get_mods() & KMOD_SHIFT:
                        world.recording_start -= 10
                    else:
                        world.recording_start -= 1
                    world.hud.notification("Recording start time is %d" % (world.recording_start))
                elif event.key == K_EQUALS and (pygame.key.get_mods() & KMOD_CTRL):
                    if pygame.key.get_mods() & KMOD_SHIFT:
                        world.recording_start += 10
                    else:
                        world.recording_start += 1
                    world.hud.notification("Recording start time is %d" % (world.recording_start))
                if isinstance(self._control, carla.VehicleControl):
                    if event.key == K_q:
                        self._control.gear = 1 if self._control.reverse else -1
                    elif event.key == K_m:
                        self._control.manual_gear_shift = not self._control.manual_gear_shift
                        self._control.gear = world.player.get_control().gear
                        world.hud.notification('%s Transmission' %
                                               ('Manual' if self._control.manual_gear_shift else 'Automatic'))
                    elif self._control.manual_gear_shift and event.key == K_COMMA:
                        self._control.gear = max(-1, self._control.gear - 1)
                    elif self._control.manual_gear_shift and event.key == K_PERIOD:
                        self._control.gear = self._control.gear + 1
                    elif event.key == K_p and not pygame.key.get_mods() & KMOD_CTRL:
                        if not self._autopilot_enabled and not sync_mode:
                            print("WARNING: You are currently in asynchronous mode and could "
                                  "experience some issues with the traffic simulation")
                        self._autopilot_enabled = not self._autopilot_enabled
                        world.player.set_autopilot(self._autopilot_enabled)
                        world.hud.notification(
                            'Autopilot %s' % ('On' if self._autopilot_enabled else 'Off'))
                    elif event.key == K_l and pygame.key.get_mods() & KMOD_CTRL:
                        current_lights ^= carla.VehicleLightState.Special1
                    elif event.key == K_l and pygame.key.get_mods() & KMOD_SHIFT:
                        current_lights ^= carla.VehicleLightState.HighBeam
                    elif event.key == K_l:
                        # Use 'L' key to switch between lights:
                        # closed -> position -> low beam -> fog
                        if not self._lights & carla.VehicleLightState.Position:
                            world.hud.notification("Position lights")
                            current_lights |= carla.VehicleLightState.Position
                        else:
                            world.hud.notification("Low beam lights")
                            current_lights |= carla.VehicleLightState.LowBeam
                        if self._lights & carla.VehicleLightState.LowBeam:
                            world.hud.notification("Fog lights")
                            current_lights |= carla.VehicleLightState.Fog
                        if self._lights & carla.VehicleLightState.Fog:
                            world.hud.notification("Lights off")
                            current_lights ^= carla.VehicleLightState.Position
                            current_lights ^= carla.VehicleLightState.LowBeam
                            current_lights ^= carla.VehicleLightState.Fog
                    elif event.key == K_i:
                        current_lights ^= carla.VehicleLightState.Interior
                    elif event.key == K_z:
                        current_lights ^= carla.VehicleLightState.LeftBlinker
                    elif event.key == K_x:
                        current_lights ^= carla.VehicleLightState.RightBlinker
                    elif event.key == K_e:
                        client.stop_recorder()
                        world.recording_enabled = False
                        print("e 눌렀음")


        if not self._autopilot_enabled:
            if isinstance(self._control, carla.VehicleControl):
                self._parse_vehicle_keys(pygame.key.get_pressed(), clock.get_time())
                self._control.reverse = self._control.gear < 0
                # Set automatic control-related vehicle lights
                if self._control.brake:
                    current_lights |= carla.VehicleLightState.Brake
                else: # Remove the Brake flag
                    current_lights &= ~carla.VehicleLightState.Brake
                if self._control.reverse:
                    current_lights |= carla.VehicleLightState.Reverse
                else: # Remove the Reverse flag
                    current_lights &= ~carla.VehicleLightState.Reverse
                if current_lights != self._lights: # Change the light state only if necessary
                    self._lights = current_lights
                    world.player.set_light_state(carla.VehicleLightState(self._lights))
            elif isinstance(self._control, carla.WalkerControl):
                self._parse_walker_keys(pygame.key.get_pressed(), clock.get_time(), world)
            world.player.apply_control(self._control)

    def _parse_vehicle_keys(self, keys, milliseconds):
        if keys[K_UP] or keys[K_w]:
            self._control.throttle = min(self._control.throttle + 0.01, 1.00)
        else:
            self._control.throttle = 0.0

        if keys[K_DOWN] or keys[K_s]:
            self._control.brake = min(self._control.brake + 0.2, 1)
        else:
            self._control.brake = 0

        steer_increment = 5e-4 * milliseconds
        if keys[K_LEFT] or keys[K_a]:
            if self._steer_cache > 0:
                self._steer_cache = 0
            else:
                self._steer_cache -= steer_increment
        elif keys[K_RIGHT] or keys[K_d]:
            if self._steer_cache < 0:
                self._steer_cache = 0
            else:
                self._steer_cache += steer_increment
        else:
            self._steer_cache = 0.0
        self._steer_cache = min(0.7, max(-0.7, self._steer_cache))
        self._control.steer = round(self._steer_cache, 1)
        self._control.hand_brake = keys[K_SPACE]

    def _parse_walker_keys(self, keys, milliseconds, world):
        self._control.speed = 0.0
        if keys[K_DOWN] or keys[K_s]:
            self._control.speed = 0.0
        if keys[K_LEFT] or keys[K_a]:
            self._control.speed = .01
            self._rotation.yaw -= 0.08 * milliseconds
        if keys[K_RIGHT] or keys[K_d]:
            self._control.speed = .01
            self._rotation.yaw += 0.08 * milliseconds
        if keys[K_UP] or keys[K_w]:
            self._control.speed = world.player_max_speed_fast if pygame.key.get_mods() & KMOD_SHIFT else world.player_max_speed
        self._control.jump = keys[K_SPACE]
        self._rotation.yaw = round(self._rotation.yaw, 1)
        self._control.direction = self._rotation.get_forward_vector()

    @staticmethod
    def _is_quit_shortcut(key):
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)


# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================


class HUD(object):
    def __init__(self, width, height):
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        font_name = 'courier' if os.name == 'nt' else 'mono'
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 12 if os.name == 'nt' else 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.help = HelpText(pygame.font.Font(mono, 16), width, height)
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()

    def on_world_tick(self, timestamp):
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, world, clock):
        self._notifications.tick(world, clock)
        if not self._show_info:
            return
        t = world.player.get_transform()
        v = world.player.get_velocity()
        c = world.player.get_control()
        compass = world.imu_sensor.compass
        heading = 'N' if compass > 270.5 or compass < 89.5 else ''
        heading += 'S' if 90.5 < compass < 269.5 else ''
        heading += 'E' if 0.5 < compass < 179.5 else ''
        heading += 'W' if 180.5 < compass < 359.5 else ''
        colhist = world.collision_sensor.get_collision_history()
        collision = [colhist[x + self.frame - 200] for x in range(0, 200)]
        max_col = max(1.0, max(collision))
        collision = [x / max_col for x in collision]
        vehicles = world.world.get_actors().filter('vehicle.*')
        self._info_text = [
            'Server:  % 16.0f FPS' % self.server_fps,
            'Client:  % 16.0f FPS' % clock.get_fps(),
            '',
            'Vehicle: % 20s' % get_actor_display_name(world.player, truncate=20),
            'Map:     % 20s' % world.map.name.split('/')[-1],
            'Simulation time: % 12s' % datetime.timedelta(seconds=int(self.simulation_time)),
            '',
            'Speed:   % 15.0f km/h' % (3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)),
            u'Compass:% 17.0f\N{DEGREE SIGN} % 2s' % (compass, heading),
            'Accelero: (%5.1f,%5.1f,%5.1f)' % (world.imu_sensor.accelerometer),
            'Gyroscop: (%5.1f,%5.1f,%5.1f)' % (world.imu_sensor.gyroscope),
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' % (t.location.x, t.location.y)),
            'GNSS:% 24s' % ('(% 2.6f, % 3.6f)' % (world.gnss_sensor.lat, world.gnss_sensor.lon)),
            'Height:  % 18.0f m' % t.location.z,
            '']
        if isinstance(c, carla.VehicleControl):
            self._info_text += [
                ('Throttle:', c.throttle, 0.0, 1.0),
                ('Steer:', c.steer, -1.0, 1.0),
                ('Brake:', c.brake, 0.0, 1.0),
                ('Reverse:', c.reverse),
                ('Hand brake:', c.hand_brake),
                ('Manual:', c.manual_gear_shift),
                'Gear:        %s' % {-1: 'R', 0: 'N'}.get(c.gear, c.gear)]
        elif isinstance(c, carla.WalkerControl):
            self._info_text += [
                ('Speed:', c.speed, 0.0, 5.556),
                ('Jump:', c.jump)]
        self._info_text += [
            '',
            'Collision:',
            collision,
            '',
            'Number of vehicles: % 8d' % len(vehicles)]
        if len(vehicles) > 1:
            self._info_text += ['Nearby vehicles:']
            distance = lambda l: math.sqrt((l.x - t.location.x)**2 + (l.y - t.location.y)**2 + (l.z - t.location.z)**2)
            vehicles = [(distance(x.get_location()), x) for x in vehicles if x.id != world.player.id]
            for d, vehicle in sorted(vehicles, key=lambda vehicles: vehicles[0]):
                if d > 200.0:
                    break
                vehicle_type = get_actor_display_name(vehicle, truncate=22)
                self._info_text.append('% 4dm %s' % (d, vehicle_type))

    def toggle_info(self):
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display):
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1.0 - y) * 30) for x, y in enumerate(item)]
                        pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                        f = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect((bar_h_offset + f * (bar_width - 6), v_offset + 8), (6, 6))
                        else:
                            rect = pygame.Rect((bar_h_offset, v_offset + 8), (f * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item:  # At this point has to be a str.
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18
        self._notifications.render(display)
        self.help.render(display)


# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================


class FadingText(object):
    def __init__(self, font, dim, pos):
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        display.blit(self.surface, self.pos)


# ==============================================================================
# -- HelpText ------------------------------------------------------------------
# ==============================================================================


class HelpText(object):
    """Helper class to handle text output using pygame"""
    def __init__(self, font, width, height):
        lines = __doc__.split('\n')
        self.font = font
        self.line_space = 18
        self.dim = (780, len(lines) * self.line_space + 12)
        self.pos = (0.5 * width - 0.5 * self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        for n, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, n * self.line_space))
            self._render = False
        self.surface.set_alpha(220)

    def toggle(self):
        self._render = not self._render

    def render(self, display):
        if self._render:
            display.blit(self.surface, self.pos)


# ==============================================================================
# -- CollisionSensor -----------------------------------------------------------
# ==============================================================================


class CollisionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        history = collections.defaultdict(int)
        for frame, intensity in self.history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        actor_type = get_actor_display_name(event.other_actor)
        self.hud.notification('Collision with %r' % actor_type)
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        self.history.append((event.frame, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)


# ==============================================================================
# -- LaneInvasionSensor --------------------------------------------------------
# ==============================================================================


class LaneInvasionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None

        # If the spawn object is not a vehicle, we cannot use the Lane Invasion Sensor
        if parent_actor.type_id.startswith("vehicle."):
            self._parent = parent_actor
            self.hud = hud
            world = self._parent.get_world()
            bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
            self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
            # We need to pass the lambda a weak reference to self to avoid circular
            # reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    @staticmethod
    def _on_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ['%r' % str(x).split()[-1] for x in lane_types]
        self.hud.notification('Crossed line %s' % ' and '.join(text))


# ==============================================================================
# -- GnssSensor ----------------------------------------------------------------
# ==============================================================================


class GnssSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.gnss')
        self.sensor = world.spawn_actor(bp, carla.Transform(carla.Location(x=1.0, z=2.8)), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: GnssSensor._on_gnss_event(weak_self, event))

    @staticmethod
    def _on_gnss_event(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude


# ==============================================================================
# -- IMUSensor -----------------------------------------------------------------
# ==============================================================================


class IMUSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.accelerometer = (0.0, 0.0, 0.0)
        self.gyroscope = (0.0, 0.0, 0.0)
        self.compass = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.imu')
        self.sensor = world.spawn_actor(
            bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda sensor_data: IMUSensor._IMU_callback(weak_self, sensor_data))

    @staticmethod
    def _IMU_callback(weak_self, sensor_data):
        self = weak_self()
        if not self:
            return
        limits = (-99.9, 99.9)
        self.accelerometer = (
            max(limits[0], min(limits[1], sensor_data.accelerometer.x)),
            max(limits[0], min(limits[1], sensor_data.accelerometer.y)),
            max(limits[0], min(limits[1], sensor_data.accelerometer.z)))
        self.gyroscope = (
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.x))),
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.y))),
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.z))))
        self.compass = math.degrees(sensor_data.compass)


# ==============================================================================
# -- RadarSensor ---------------------------------------------------------------
# ==============================================================================


class RadarSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        bound_x = 0.5 + self._parent.bounding_box.extent.x
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        bound_z = 0.5 + self._parent.bounding_box.extent.z

        self.velocity_range = 7.5 # m/s
        world = self._parent.get_world()
        self.debug = world.debug
        bp = world.get_blueprint_library().find('sensor.other.radar')
        bp.set_attribute('horizontal_fov', str(35))
        bp.set_attribute('vertical_fov', str(20))
        self.sensor = world.spawn_actor(
            bp,
            carla.Transform(
                carla.Location(x=bound_x + 0.05, z=bound_z+0.05),
                carla.Rotation(pitch=5)),
            attach_to=self._parent)
        # We need a weak reference to self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda radar_data: RadarSensor._Radar_callback(weak_self, radar_data))

    @staticmethod
    def _Radar_callback(weak_self, radar_data):
        self = weak_self()
        if not self:
            return
        # To get a numpy [[vel, altitude, azimuth, depth],...[,,,]]:
        # points = np.frombuffer(radar_data.raw_data, dtype=np.dtype('f4'))
        # points = np.reshape(points, (len(radar_data), 4))

        current_rot = radar_data.transform.rotation
        for detect in radar_data:
            azi = math.degrees(detect.azimuth)
            alt = math.degrees(detect.altitude)
            # The 0.25 adjusts a bit the distance so the dots can
            # be properly seen
            fw_vec = carla.Vector3D(x=detect.depth - 0.25)
            carla.Transform(
                carla.Location(),
                carla.Rotation(
                    pitch=current_rot.pitch + alt,
                    yaw=current_rot.yaw + azi,
                    roll=current_rot.roll)).transform(fw_vec)

            def clamp(min_v, max_v, value):
                return max(min_v, min(value, max_v))

            norm_velocity = detect.velocity / self.velocity_range # range [-1, 1]
            r = int(clamp(0.0, 1.0, 1.0 - norm_velocity) * 255.0)
            g = int(clamp(0.0, 1.0, 1.0 - abs(norm_velocity)) * 255.0)
            b = int(abs(clamp(- 1.0, 0.0, - 1.0 - norm_velocity)) * 255.0)
            self.debug.draw_point(
                radar_data.transform.location + fw_vec,
                size=0.075,
                life_time=0.06,
                persistent_lines=False,
                color=carla.Color(r, g, b))

# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================


class CameraManager(object):
    def __init__(self, parent_actor, hud, gamma_correction):
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.recording = False
        bound_x = 0.5 + self._parent.bounding_box.extent.x
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        bound_z = 0.5 + self._parent.bounding_box.extent.z
        Attachment = carla.AttachmentType

        if not self._parent.type_id.startswith("walker.pedestrian"):
            self._camera_transforms = [
                (carla.Transform(carla.Location(x=-2.0*bound_x, y=+0.0*bound_y, z=2.0*bound_z), carla.Rotation(pitch=8.0)), Attachment.SpringArm),
                (carla.Transform(carla.Location(x=+0.8*bound_x, y=+0.0*bound_y, z=1.3*bound_z)), Attachment.Rigid),
                (carla.Transform(carla.Location(x=+1.9*bound_x, y=+1.0*bound_y, z=1.2*bound_z)), Attachment.SpringArm),
                (carla.Transform(carla.Location(x=-2.8*bound_x, y=+0.0*bound_y, z=4.6*bound_z), carla.Rotation(pitch=6.0)), Attachment.SpringArm),
                (carla.Transform(carla.Location(x=-1.0, y=-1.0*bound_y, z=0.4*bound_z)), Attachment.Rigid)]
        else:
            self._camera_transforms = [
                (carla.Transform(carla.Location(x=-2.5, z=0.0), carla.Rotation(pitch=-8.0)), Attachment.SpringArm),
                (carla.Transform(carla.Location(x=1.6, z=1.7)), Attachment.Rigid),
                (carla.Transform(carla.Location(x=2.5, y=0.5, z=0.0), carla.Rotation(pitch=-8.0)), Attachment.SpringArm),
                (carla.Transform(carla.Location(x=-4.0, z=2.0), carla.Rotation(pitch=6.0)), Attachment.SpringArm),
                (carla.Transform(carla.Location(x=0, y=-2.5, z=-0.0), carla.Rotation(yaw=90.0)), Attachment.Rigid)]

        self.transform_index = 1
        self.sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB', {}],
            ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)', {}],
            ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)', {}],
            ['sensor.camera.depth', cc.LogarithmicDepth, 'Camera Depth (Logarithmic Gray Scale)', {}],
            ['sensor.camera.semantic_segmentation', cc.Raw, 'Camera Semantic Segmentation (Raw)', {}],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette, 'Camera Semantic Segmentation (CityScapes Palette)', {}],
            ['sensor.camera.instance_segmentation', cc.CityScapesPalette, 'Camera Instance Segmentation (CityScapes Palette)', {}],
            ['sensor.camera.instance_segmentation', cc.Raw, 'Camera Instance Segmentation (Raw)', {}],
            ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)', {'range': '50'}],
            ['sensor.camera.dvs', cc.Raw, 'Dynamic Vision Sensor', {}],
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB Distorted',
                {'lens_circle_multiplier': '3.0',
                'lens_circle_falloff': '3.0',
                'chromatic_aberration_intensity': '0.5',
                'chromatic_aberration_offset': '0'}],
            ['sensor.camera.optical_flow', cc.Raw, 'Optical Flow', {}],
        ]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            bp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                bp.set_attribute('image_size_x', str(hud.dim[0]))
                bp.set_attribute('image_size_y', str(hud.dim[1]))
                if bp.has_attribute('gamma'):
                    bp.set_attribute('gamma', str(gamma_correction))
                for attr_name, attr_value in item[3].items():
                    bp.set_attribute(attr_name, attr_value)
            elif item[0].startswith('sensor.lidar'):
                self.lidar_range = 50

                for attr_name, attr_value in item[3].items():
                    bp.set_attribute(attr_name, attr_value)
                    if attr_name == 'range':
                        self.lidar_range = float(attr_value)

            item.append(bp)
        self.index = None

    def toggle_camera(self):
        self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
        self.set_sensor(self.index, notify=False, force_respawn=True)

    def set_sensor(self, index, notify=True, force_respawn=False):
        index = index % len(self.sensors)
        needs_respawn = True if self.index is None else \
            (force_respawn or (self.sensors[index][2] != self.sensors[self.index][2]))
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                self._camera_transforms[self.transform_index][0],
                attach_to=self._parent,
                attachment_type=self._camera_transforms[self.transform_index][1])
            # We need to pass the lambda a weak reference to self to avoid
            # circular reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))
        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index

    def next_sensor(self):
        self.set_sensor(self.index + 1)

    def toggle_recording(self):
        self.recording = not self.recording
        self.hud.notification('Recording %s' % ('On' if self.recording else 'Off'))

    def render(self, display):
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        if self.sensors[self.index][0].startswith('sensor.lidar'):
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 4), 4))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self.hud.dim) / (2.0 * self.lidar_range)
            lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
            lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3)
            lidar_img = np.zeros((lidar_img_size), dtype=np.uint8)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self.surface = pygame.surfarray.make_surface(lidar_img)
        elif self.sensors[self.index][0].startswith('sensor.camera.dvs'):
            # Example of converting the raw_data from a carla.DVSEventArray
            # sensor into a NumPy array and using it as an image
            dvs_events = np.frombuffer(image.raw_data, dtype=np.dtype([
                ('x', np.uint16), ('y', np.uint16), ('t', np.int64), ('pol', np.bool)]))
            dvs_img = np.zeros((image.height, image.width, 3), dtype=np.uint8)
            # Blue is positive, red is negative
            dvs_img[dvs_events[:]['y'], dvs_events[:]['x'], dvs_events[:]['pol'] * 2] = 255
            self.surface = pygame.surfarray.make_surface(dvs_img.swapaxes(0, 1))
        elif self.sensors[self.index][0].startswith('sensor.camera.optical_flow'):
            image = image.get_color_coded_flow()
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        else:
            image.convert(self.sensors[self.index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
            #############minhong#########################################
            global my_img
            my_img = array.copy()
        if self.recording:
            #image.save_to_disk('_out/%08d' % image.frame)
            image.save_to_disk('~/opt/carla-simulator/PythonAPI/examples/recording/%08d' % image.frame)


# ==============================================================================
# minhong
# ==============================================================================
class turnState(Enum):
    FORWARD = 1
    LEFT = 2
    RIGHT = 3


global saturation_thr,x_grad_thr,red_thr,green_thr,blue_thr
saturation_thr = [10,255]
x_grad_thr = [0,255]
red_thr = [0,255]
green_thr = [0,255]
blue_thr = [190,255] 

global birdeye_top,birdeye_bottom,birdeye_top_width,birdeye_bottom_width,birdeye_width
birdeye_top = 420
birdeye_bottom =  540
birdeye_top_width = 3
birdeye_bottom_width = 10
birdeye_width = 20

global image_width,image_height
image_width = 320
image_height = 240

global turn_state
turn_state = turnState.FORWARD
def find_firstfit(binary_array, direction, lower_threshold=20):
    # direction : -1 (right to left), 1 (left to right)
    start, end = (binary_array.shape[0]-1, -1) if direction == -1 else (0, binary_array.shape[0])
    for i in range(start, end, direction):
        # If value is larger than threshold, it is white image
        if binary_array[i] > lower_threshold:
            return i
    return -1

def color_gradient_filter(img):
    filter_thr_dict = [] 
    s_thresh = saturation_thr
    sx_thresh = x_grad_thr
    r_thresh = red_thr
    g_thresh = green_thr
    b_thresh = blue_thr
    X_M_PER_PIX = 3.7/70
    Y_M_PER_PIX = 30/720
    img = np.copy(img)
    R = img[:,:,0]
    G = img[:,:,1]
    B = img[:,:,2]
    
    # Threshold red channel
    rbinary = np.zeros_like(R)
    rbinary[(R >= r_thresh[0]) & (R <= r_thresh[1])] = 1
    # 

    gbinary = np.zeros_like(G)
    gbinary[(G >= g_thresh[0]) & (G <= g_thresh[1])] = 1
    bbinary = np.zeros_like(B)
    bbinary[(B >= b_thresh[0]) & (B <= b_thresh[1])] = 1

    # Convert to HLS color space and separate the s channel
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)#.astype(np.float)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h_channel = hls[:,:,0]
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx)) #cv2.convertScaleAbs(abs_sobelx)

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Threshold saturation channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    
    combined_binary = np.zeros_like(sxbinary)
    # combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
    # combined_binary[(s_binary == 1) | (sxbinary == 1) & (rbinary == 1)] = 1
    
    combined_binary[(s_binary == 1) & (sxbinary == 1) & (rbinary == 1) & (gbinary == 1) & (bbinary == 1)] = 1
    
    return combined_binary


def birdeye_warp(img):
            x_size = img.shape[1]
            y_size = img.shape[0]
            x_mid = x_size/2

            # Tunable Parameter
            top = birdeye_top
            bottom = birdeye_bottom
            top_margin = birdeye_top_width
            bottom_margin = birdeye_bottom_width
            birdeye_margin = birdeye_width

            # 4 Source coordinates
            src1 = [x_mid + top_margin, top] # top_right
            src2 = [x_mid + bottom_margin, bottom] # bottom_right
            src3 = [x_mid - bottom_margin, bottom] # bottom_left
            src4 = [x_mid - top_margin, top] # top_left
            src = np.float32([src1, src2, src3, src4])

            # 4 destination coordinates
            dst1 = [x_mid + birdeye_margin, 0]
            dst2 = [x_mid + birdeye_margin, y_size]
            dst3 = [x_mid - birdeye_margin, y_size]
            dst4 = [x_mid - birdeye_margin, 0]
            dst = np.float32([dst1, dst2, dst3, dst4])

            # Given src and dst points, calculate the perspective transform matrix
            trans_matrix = cv2.getPerspectiveTransform(src, dst)
            # Warp the image using OpenCV warpPerspective()
            img_size = (img.shape[1],img.shape[0])
            birdeye_warped = cv2.warpPerspective(img, trans_matrix, img_size)

            # Return the resulting image and matrix
            return birdeye_warped



prev_left_x = -1    
prev_right_x = -1
prev_left_slope = None
prev_right_slope = None

def calculate_sliding_window(img):
    
    global prev_left_x
    global prev_right_x
    global prev_left_slope
    global prev_right_slope

     
    ##### Tunable parameter
    windows_num = 24
    window_width = 10
    # The part recognized from both edges is ignored
    x_margin = 30
    consecutive_y_margin = 100
    # The x of succesive sliding windows should not differ by more than this value
    noise_threshold = 30
    # number of sliding window should larger than window_threshold
    window_threshold = 2
    # curve_threshold = 15
    slope_threshold = 20

    out_img = np.dstack((img,img,img))*255
    window_height = np.int(img.shape[0]/windows_num)

    consecutive_y_idx = consecutive_y_margin // window_height
    x_mid = np.int(img.shape[1]/2)
    # lw_arr : left sliding window array's position
    # rw_arr : right sliding window array's position
    lw_arr = []
    rw_arr = []
    for window_idx in range(windows_num):



        window_top = np.int(window_height * (windows_num - window_idx - 1))
        window_bottom = np.int(window_height * (windows_num - window_idx))

        leftx = find_firstfit(np.sum(img[window_top:window_bottom,:x_mid], axis=0), -1, window_height // 2)
        rightx = find_firstfit(np.sum(img[window_top:window_bottom,x_mid:], axis=0), 1, window_height // 2)
        if leftx > x_margin and leftx < x_mid - x_margin:

            if not lw_arr:

                # if prev_left_x == -1 or abs(prev_left_x - leftx) < noise_threshold:
                lw_arr.append((leftx, window_idx))
                cv2.rectangle(out_img,(leftx-window_width,window_bottom),(leftx,window_top),(0,255,0), 2)
            elif (window_idx - lw_arr[-1][1]) > consecutive_y_idx:

                pass
            elif abs(lw_arr[-1][0] - leftx) < noise_threshold:
                lw_arr.append((leftx, window_idx))
                cv2.rectangle(out_img,(leftx-window_width,window_bottom),(leftx,window_top),(0,255,0), 2)
            if rightx > x_margin and rightx < x_mid - x_margin:
                if not rw_arr:

                    # if prev_right_x == -1 or abs(prev_right_x - (rightx+x_mid)) < noise_threshold:
                    rw_arr.append((rightx + x_mid, window_idx))
                    cv2.rectangle(out_img,(rightx + x_mid,window_bottom),(rightx + x_mid + window_width,window_top),(0,255,0), 2)
                elif (window_idx - rw_arr[-1][1]) > consecutive_y_idx:

                    pass
                elif abs(rw_arr[-1][0] - (rightx + x_mid)) < noise_threshold:

                    rw_arr.append((rightx + x_mid, window_idx))
                    cv2.rectangle(out_img,(rightx + x_mid,window_bottom),(rightx + x_mid + window_width,window_top),(0,255,0), 2)

        ### Fit a first order polynomial to each sliding windows
        isLeftValid = len(lw_arr) >= window_threshold
        isRightValid = len(rw_arr) >= window_threshold
        left_slope = -1
        right_slope = -1
        first_left_x_margin = 0
        first_right_x_margin = 0

        if isLeftValid:
            try:
                left_slope = math.degrees(math.atan(np.polyfit([x for (x, y) in lw_arr], [y * window_height for (x, y) in lw_arr], 1)[0]))
                if left_slope > 0:
                    left_slope = 90 - left_slope
                elif left_slope < 0:
                    left_slope = -90 - left_slope

                if left_slope < -slope_threshold:
                    isLeftValid = False
                
            except:
                isLeftValid = False

        if isRightValid:
            try:
                right_slope = math.degrees(math.atan(np.polyfit([x for (x, y) in rw_arr], [y * window_height for (x, y) in rw_arr], 1)[0]))
                if right_slope > 0:
                    right_slope = 90 - right_slope
                elif right_slope < 0:
                    right_slope = -90 - right_slope

                if right_slope > slope_threshold:
                    isRightValid = False
            except:
                isRightValid = False

        left_idx = 0
        right_idx = 0
        if isLeftValid and isRightValid:
            while left_idx < len(lw_arr) and right_idx < len(rw_arr) and lw_arr[left_idx][1] != rw_arr[right_idx][1]:
                if lw_arr[left_idx][1] > rw_arr[right_idx][1]:
                        right_idx += 1
                else:
                    left_idx += 1
                if left_idx < len(lw_arr) and right_idx < len(rw_arr):
                    first_left_x_margin = lw_arr[left_idx][0]
                    first_right_x_margin = img.shape[1] - rw_arr[right_idx][0]
        elif isLeftValid:
            first_left_x_margin = lw_arr[0][0]
        elif isRightValid:
            first_right_x_margin = img.shape[1] - rw_arr[0][0]
    
        if left_idx == len(lw_arr):
            isLeftValid = False
        if right_idx == len(rw_arr):
            isRightValid = False
    
        if isLeftValid:
            prev_left_x = lw_arr[0][0]
            prev_left_slope = left_slope
        if isRightValid:
            prev_right_x = rw_arr[0][0]
            prev_right_slope = right_slope
    return out_img, left_slope, right_slope,isLeftValid,isRightValid, first_left_x_margin,first_right_x_margin









class lane_keeping_module(object) :
    def __init__(self):
            self.turn_state = turnState.LEFT
            self.image_width = 320
            self.image_height = 240
            #if self.debug_window:
            self.trackbar_img = np.zeros((1,400),np.uint8)
            cv2.namedWindow('original_image')
            cv2.namedWindow('birdeye_image')
            cv2.namedWindow('sliding_window')
            cv2.namedWindow('filtered_birdeye')
            cv2.namedWindow('TrackBar')
                        
            cv2.moveWindow('original_image',    350*0,  350*0)
            cv2.moveWindow('birdeye_image',     350*0,  350*1)
            cv2.moveWindow('sliding_window',    350*1,  350*0)
            cv2.moveWindow('filtered_birdeye',  350*1,  350*1)
            cv2.moveWindow('TrackBar',          350*2,  350*0)

            # Create Trackbar
            
            cv2.createTrackbar("[clr]x_grad_min", "TrackBar", x_grad_thr[0], 255, self.onChange)
            cv2.createTrackbar("[clr]x_grad_max", "TrackBar", x_grad_thr[1], 255, self.onChange)
            cv2.createTrackbar("[clr]sat_min", "TrackBar", saturation_thr[0], 255, self.onChange)
            cv2.createTrackbar("[clr]sat_max", "TrackBar", saturation_thr[1], 255,self.onChange)
            cv2.createTrackbar("[clr]red_min", "TrackBar", red_thr[0], 255, self.onChange)
            cv2.createTrackbar("[clr]red_max", "TrackBar", red_thr[1], 255, self.onChange)
            cv2.createTrackbar("[clr]green_min", "TrackBar", green_thr[0], 255, self.onChange)
            cv2.createTrackbar("[clr]green_max", "TrackBar", green_thr[1], 255, self.onChange)
            cv2.createTrackbar("[clr]blue_min", "TrackBar",blue_thr[0], 255, self.onChange)
            cv2.createTrackbar("[clr]blue_max", "TrackBar",blue_thr[1], 255, self.onChange)
            cv2.createTrackbar("[be]top", "TrackBar", birdeye_top, 240, self.onChange)
            cv2.createTrackbar("[be]bottom", "TrackBar",birdeye_bottom, 240, self.onChange)
            cv2.createTrackbar("[be]top_width", "TrackBar",birdeye_top_width, 320, self.onChange)
            cv2.createTrackbar("[be]bottom_width", "TrackBar",birdeye_bottom_width, 320, self.onChange)
            cv2.createTrackbar("[be]birdeye_width", "TrackBar",birdeye_width, 320, self.onChange)        

    def onChange(self,pos):
        pass

    def calculate_velocity_and_angle(self,ls,rs,lv,rv,lx,rx):
        global turn_state
        left_ground_truth = 3.8
        right_ground_truth = -3.8
        steer_sensitivity = 0.001
        target_angle = 0
        if turn_state == turnState.LEFT:
            if(lv and ls>0) :
                
                turn_state = turnState.FORWARD
            if rv :
                target_angle = (rx - self.image_height / 6) / 10
            else:
                target_angle = left_ground_truth
                if(target_angle == 0) :
                    turn_state = turnState.FORWARD
            
        elif turn_state == turnState.RIGHT:
            if(rv and rs <0):
                turn_state = turnState.FORWARD
            if lv:
                target_angle = (self.image_height/6 - lx) / 10
            else:
                target_angle = right_ground_truth
            
        else:
            if lv and rv :
                #print("rs : ",rs)
                #print("ls : ",ls)
                if ls<right_ground_truth and rs < right_ground_truth*3:
                    turn_state = turnState.LEFT
                if rs>left_ground_truth and ls > left_ground_truth *3:
                    turn_state = turnState.RIGHT
                target_angle = (rx-lx)/10
            elif rv:
                target_angle = (rx - self.image_height / 8) /10
            elif lv:
                target_angle = (self.image_height/8 - lx) / 10
            
        if lv:
            #print('left:',ls,lx)
            pass
        if rv:
            #print('right:',rs,rx)
            pass

            
        target_angle = target_angle * steer_sensitivity 
        #return target_angle
        return turn_state,target_angle
    def image_callback(self,my_img):
         
        x_grad_thr[0] = cv2.getTrackbarPos("[clr]x_grad_min", "TrackBar")
        x_grad_thr[1] = cv2.getTrackbarPos("[clr]x_grad_max", "TrackBar")
        saturation_thr[0] = cv2.getTrackbarPos("[clr]sat_min", "TrackBar")
        saturation_thr[1] = cv2.getTrackbarPos("[clr]sat_max", "TrackBar")
        red_thr[0] = cv2.getTrackbarPos("[clr]red_min", "TrackBar")
        red_thr[1] = cv2.getTrackbarPos("[clr]red_max", "TrackBar")
        green_thr[0] = cv2.getTrackbarPos("[clr]green_min", "TrackBar")
        green_thr[1] = cv2.getTrackbarPos("[clr]green_max", "TrackBar")
        blue_thr[0] = cv2.getTrackbarPos("[clr]blue_min", "TrackBar")
        blue_thr[1] = cv2.getTrackbarPos("[clr]blue_max", "TrackBar")
        birdeye_top = cv2.getTrackbarPos("[be]top", "TrackBar")
        birdeye_bottom = cv2.getTrackbarPos("[be]bottom", "TrackBar")
        birdeye_top_width = cv2.getTrackbarPos("[be]top_width", "TrackBar")
        birdeye_bottom_width = cv2.getTrackbarPos("[be]bottom_width", "TrackBar")
        birdeye_width = cv2.getTrackbarPos("[be]birdeye_width", "TrackBar")

        cv2.waitKey(1)
        birdeye_warped = birdeye_warp(my_img)
        filtered_birdeye = color_gradient_filter(birdeye_warped)
        sliding_window, ls, rs, lv, rv, lx, rx = calculate_sliding_window(filtered_birdeye)
        cv2.imshow('TrackBar',self.trackbar_img)
        cv2.imshow('original_image',my_img)
        cv2.imshow('birdeye_image',birdeye_warped)
        cv2.imshow('sliding_window',sliding_window)
        cv2.imshow('filtered_birdeye',(filtered_birdeye*255).astype(np.uint8))
        state,target_angle = self.calculate_velocity_and_angle(ls,rs,lv,rv,lx,rx)
        return state,target_angle





# ==============================================================================
# -- game_loop() ---------------------------------------------------------------
# ==============================================================================


def game_loop(args):
    pygame.init()
    pygame.font.init()
    world = None
    original_settings = None
    global lane_keeping_module
    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(20.0)

        sim_world = client.load_world('TOWN04')
        #sim_world = client.get_world()
        if args.sync:
            original_settings = sim_world.get_settings()
            settings = sim_world.get_settings()
            if not settings.synchronous_mode:
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = 0.05
            sim_world.apply_settings(settings)

            traffic_manager = client.get_trafficmanager()
            traffic_manager.set_synchronous_mode(True)

        if args.autopilot and not sim_world.get_settings().synchronous_mode:
            print("WARNING: You are currently in asynchronous mode and could "
                  "experience some issues with the traffic simulation")

        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)
        display.fill((0,0,0))
        pygame.display.flip()
        
        hud = HUD(args.width, args.height)
        world = World(sim_world, hud, args)
        controller = KeyboardControl(world, args.autopilot)
        lane_keeping_module = lane_keeping_module()
        print("Recording on file: %s" % client.start_recorder(args.recorder_filename))
        if (args.recorder_time > 0):
            time.sleep(args.recorder_time)
    
        if args.sync:
            sim_world.tick()
        else:
            sim_world.wait_for_tick()

        clock = pygame.time.Clock()
        world.player._control = carla.VehicleControl()
        world.player._control.throttle = 0.0
        world.player._steer_cache = 0.0
        target_angle = 0.0
        while True:
            if args.sync:
                sim_world.tick()
            clock.tick_busy_loop(60)
            if controller.parse_events(client, world, clock, args.sync):
                return
            world.tick(clock)
            world.render(display)
            pygame.display.flip()

            ########### minhong #######################################

            global x_size
            global birdeye_top
            global birdeye_bottom
            global birdeye_top_width
            global birdeye_bottom_width
            global birdeye_width
            global flag
            world.player.apply_control(world.player._control)
            last_angle = target_angle
            state,target_angle = lane_keeping_module.image_callback(my_img)

            fl = carla.Vehicle.get_wheel_steer_angle(world.player,carla.VehicleWheelLocation.FL_Wheel)
            fr = carla.Vehicle.get_wheel_steer_angle(world.player,carla.VehicleWheelLocation.FR_Wheel)
            if(turn_state == turnState.FORWARD):
                world.player._control.throttle = min(world.player._control.throttle + 0.06,0.55) #0.6
            else :
                if(target_angle < -0.03):
                    world.player._control.throttle = min(world.player._control.throttle + 0.05,0.5)
                if(-0.03<target_angle < 0):
                    world.player._control.throttle = max(world.player._control.throttle - 0.1,0.4)

            steer_increment = 5e-4 * 6
            if(turn_state == turnState.LEFT) :
                if(world.player._steer_cache >0):
                    world.player._steer_cache = 0
                else :
                    world.player._steer_cache -= steer_increment
                
                if(world.player._control.steer < -target_angle and target_angle < 0.028):
                    print("too much left")
                    world.player._steer_cache = -world.player._steer_cache*2
                    world.player._control.steer = -target_angle*2
                
                elif(world.player._control.steer >  target_angle):
                    world.player.turn_state = turnState.FORWARD
                    print("steer calibration")
                    world.player._steer_cache = -world.player._steer_cache*2
                    world.player._control.steer = round(world.player._steer_cache,1)
                else :
                    world.player._control.steer = -target_angle

            
            elif (turn_state == turnState.RIGHT):
                if(world.player._steer_cache <0 and flag !=1):
                    world.player._steer_cache = 0
                else:
                    world.player._steer_cache += steer_increment
                if(world.player._control.steer > -target_angle and target_angle > -0.028):
                    print("too much right")
                    world.player._steer_cache = -world.player._steer_cache* 1.2
                    flag = 1

                    #world.player._control.steer = -target_angle * 2.2
                    world.player._control.steer = round(world.player._steer_cache,1)

                elif(world.player._control.steer < target_angle):  
                    world.player.turn_state = turnState.FORWARD
                    print("steer calibration")
                    world.player._steer_cache = -world.player._steer_cache*2
                    world.player._control.steer = round(world.player._steer_cache,1)
                else:
                    world.player._control.steer = -target_angle
                    flag = 0

                    
            
            else:
                #world.player._steer_cache = 0.0
                if(turn_state == turnState.FORWARD):
                    if(world.player._control.steer > -target_angle and target_angle > -0.028):
                        #world.player._steer_cache = -world.player._steer_cache* 1.5
                        #world.player._control.steer = round(world.player._steer_cache,1)
                        world.player._control.steer = -target_angle * 0.2

                    elif(world.player._control.steer < -target_angle and target_angle < 0.028):
                        #world.player._steer_cache = -world.player._steer_cache*1.5
                        #world.player._control.steer = round(world.player._steer_cache,1)
                        world.player._control.steer = -target_angle * 0.2

                    
       
                else:
                    world.player._steer_cache = 0
                
                #world.player._control.steer = round(world.player._steer_cache,1)


            

            world.player._steer_cache = min(0.7,max(-0.7,world.player._steer_cache))
            #world.player._control.steer = round(world.player._steer_cache,1)
            #world.player._control.steer = target_angle
            #if(world.player._control.steer != 0):
                #print("ratio :",target_angle/world.player._control.steer)
            print("state:",turn_state)
            print("target_angle:",target_angle)
            print("current_steer:",world.player._control.steer)
            print("-----------------------------")
            


    
    finally:

        if original_settings:
            sim_world.apply_settings(original_settings)

        if (world and world.recording_enabled):
            client.stop_recorder()

        if world is not None:
            world.destroy()
        
        
        print("Stop recording")
        client.stop_recorder()

        pygame.quit()


# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================


def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='window resolution (default: 1280x720)')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.*',
        help='actor filter (default: "vehicle.*")')
    argparser.add_argument(
        '--generation',
        metavar='G',
        default='2',
        help='restrict to certain actor generation (values: "1","2","All" - default: "2")')
    argparser.add_argument(
        '--rolename',
        metavar='NAME',
        default='hero',
        help='actor role name (default: "hero")')
    argparser.add_argument(
        '--gamma',
        default=2.2,
        type=float,
        help='Gamma correction of the camera (default: 2.2)')
    argparser.add_argument(
        '--sync',
        action='store_true',
        help='Activate synchronous mode execution')
    argparser.add_argument(
        '-f', '--recorder_filename',
        metavar='F',
        default="test1.log",
        help='recorder filename (test1.log)')
    argparser.add_argument(
        '-t', '--recorder_time',
        metavar='T',
        default=0,
        type=int,
        help='recorder duration (auto-stop)')
    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    try:

        game_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':

    main()
