import glob
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
from carla import ColorConverter as cc

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
    from pygame.locals import K_c
    from pygame.locals import K_g
    from pygame.locals import K_d
    from pygame.locals import K_h
    from pygame.locals import K_m
    from pygame.locals import K_n
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_r
    from pygame.locals import K_s
    from pygame.locals import K_w
    from pygame.locals import K_l
    from pygame.locals import K_i
    from pygame.locals import K_z
    from pygame.locals import K_x
    from pygame.locals import K_MINUS
    from pygame.locals import K_EQUALS
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

import random
import time
import numpy as np
import cv2
import sys

IMG_HEIGHT = 480
IMG_WIDTH = 640

DIM = (IMG_WIDTH, IMG_HEIGHT)



class MainEnv:

    IMG_HEIGHT = 480
    IMG_WIDTH = 640
    Attachment = carla.AttachmentType

    def __init__(self, world, trainDataType):
        self.world = world
        self.blueprint_library = world.get_blueprint_library()
        self.actor_list = []
        self.surface = None
        self._control = None
        self.trainDataType = trainDataType

        # Main Actor
        bp = random.choice(self.blueprint_library.filter('mustang'))
        transform = random.choice(self.world.get_map().get_spawn_points())
        self.vehicle = self.world.spawn_actor(bp, transform)
        self.actor_list.append(self.vehicle)

        # Main View Camera    
        camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(self.IMG_WIDTH))
        camera_bp.set_attribute('image_size_y', str(self.IMG_HEIGHT))
        camera_transform = carla.Transform(carla.Location(x=-4, y=0, z=2))
        camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle, attachment_type=self.Attachment.SpringArm)
        self.actor_list.append(camera)
        camera.listen(lambda image: self.mainSurface(image))

        # Training Image Camera
        if trainDataType=='lidar':
            sensor_bp = self.blueprint_library.find('sensor.lidar.ray_cast')
            sensor_bp.set_attribute('range', '50')
            sensor_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
            sensor = self.world.spawn_actor(sensor_bp, sensor_transform, attach_to=self.vehicle)
            self.actor_list.append(sensor)
            sensor.listen(lambda image: self.saveTrain(image))
        elif trainDataType=='rgb':
            sensor_bp = self.blueprint_library.find('sensor.camera.rgb')
            sensor_bp.set_attribute('image_size_x', str(self.IMG_WIDTH))
            sensor_bp.set_attribute('image_size_y', str(self.IMG_HEIGHT))
            sensor_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
            sensor = self.world.spawn_actor(sensor_bp, sensor_transform, attach_to=self.vehicle, attachment_type=self.Attachment.SpringArm)
            self.actor_list.append(sensor)
            camera.listen(lambda image: self.saveTrain(image)) 
        else:
            print("SAVE TRAINING DATA TYPE NOT SUPPORTED!")                   

    def applyControl(self, _control):
        self._control = _control
        self.vehicle.apply_control(_control)

    def mainSurface(self, image):
        image.convert(cc.Raw)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1)) 

    def saveTrain(self, image):
        if self._control == None:
            return
        
        throttle = self._control.throttle
        steering = self._control.steer
        brake = self._control.brake    
        
        if self.trainDataType=='lidar':    
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 3), 3))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(DIM) / 100.0
            lidar_data += (0.5 * DIM[0], 0.5 * DIM[1])
            lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (DIM[0], DIM[1], 3)
            lidar_img = np.zeros((lidar_img_size), dtype = int)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            final_image = lidar_img
        
        elif self.trainDataType=='rgb':
            image.convert(cc.Raw)
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            final_image = array

        print(np.array([final_image,[throttle, steering, brake]]))    


    def render(self, display):
        if self.surface is not None:
            display.blit(self.surface, (0, 0))   

    def endEnv(self):
        for actor in self.actor_list:
            if actor is not None:
                actor.destroy()

class Controller:

    def __init__(self, world):
        self._control = carla.VehicleControl()
        world.vehicle.set_autopilot(False)

    def parse_events(self, client, world, clock):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True
                elif event.key == K_r and not (pygame.key.get_mods() & KMOD_CTRL):
                    world.camera_manager.toggle_recording()
                if isinstance(self._control, carla.VehicleControl):
                    if event.key == K_q:
                        self._control.gear = 1 if self._control.reverse else -1
                    elif event.key == K_m:
                        self._control.manual_gear_shift = not self._control.manual_gear_shift
                        self._control.gear = world.vehicle.get_control().gear
                        world.hud.notification('%s Transmission' %
                                               ('Manual' if self._control.manual_gear_shift else 'Automatic'))
                    elif self._control.manual_gear_shift and event.key == K_COMMA:
                        self._control.gear = max(-1, self._control.gear - 1)
                    elif self._control.manual_gear_shift and event.key == K_PERIOD:
                        self._control.gear = self._control.gear + 1

        if isinstance(self._control, carla.VehicleControl):
            self._parse_vehicle_keys(pygame.key.get_pressed(), clock.get_time())
            self._control.reverse = self._control.gear < 0
        world.applyControl(self._control)

    def _parse_vehicle_keys(self, keys, milliseconds):
        if keys[K_UP] or keys[K_w]:
            self._control.throttle = min(self._control.throttle + 0.01, 1)
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

    @staticmethod
    def _is_quit_shortcut(key):
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)    

def main():

    world = None

    try:

        client = carla.Client('127.0.0.1', 2000)
        client.set_timeout(2.0)

        display = pygame.display.set_mode(
            (IMG_WIDTH,IMG_HEIGHT),
            pygame.HWSURFACE | pygame.DOUBLEBUF)
  
        world = MainEnv(client.get_world(), 'lidar') # lidar or rgb
        controller = Controller(world)

        clock = pygame.time.Clock()
        
        while True:
            if controller.parse_events(client, world, clock):
                return
            clock.tick_busy_loop(60)
            world.render(display)
            pygame.display.flip()

        time.sleep(4)

    finally:
        if world is not None:
            world.endEnv()
        pygame.quit()

if __name__ == '__main__':

    main()
