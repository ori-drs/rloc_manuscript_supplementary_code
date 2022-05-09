# Object Desired Dimensions
#
# For plank:
# height <= 0.5,
# roll in [0, np.pi / 4],
# pitch in [-np.pi / 4, np.pi / 4],
# yaw in [-np.pi / 4, np.pi / 4] (opp. sign of pitch)
#
# For stairs:
# yaw in [-np.pi / 4, np.pi / 4]
# height <= 1
#
# For Uneven Terrain:
# height <= 0.05 (prefer to keep it 0.05)
# smoothing in [6, 10]
#
# For Sine Terrain:
# height <= 0.1
# period in [np.pi / 2, 3 * np.pi / 2]
#
# For Brick Terrain:
# height in [0.04, 0.1]


import os
import random

from tqdm import tqdm

import numpy as np

from common.objects import World
from common.paths import HEIGHT_MAPS_PATH, CURRENT_DATETIME_STR
from common.visualizer import show_height_map


def main():
    visualize = False

    save_as_png = True

    if save_as_png:
        os.makedirs(HEIGHT_MAPS_PATH + CURRENT_DATETIME_STR, exist_ok=True)

    file_name_prefix = 'scene'

    scenes_to_generate = 1000

    available_objects = ['plank', 'stairs', 'uneven_terrain', 'sine_wave_terrain', 'brick_terrain']

    object_probability = [0.2, 0.3, 0.1, 0.1, 0.3]

    world_length = 1001
    world_width = 1001

    for scene in tqdm(range(scenes_to_generate)):
        world = World(world_length, world_width, 0)
        number_of_objects = np.random.randint(3, 6)

        objects = random.choices(population=range(len(available_objects)), weights=object_probability,
                                 k=number_of_objects)
        objects.sort()
        objects = [available_objects[obj] for obj in objects]
        objects.reverse()

        pos_limit_len = np.arange(number_of_objects) * number_of_objects * 50
        pos_limit_width = np.arange(number_of_objects) * number_of_objects * 50

        if number_of_objects == 1:
            pos_limit_len = np.random.normal(np.array([world_length / 2]), 50).astype(int)
            pos_limit_width = np.random.normal(np.array([world_length / 2]), 50).astype(int)

        elif number_of_objects == 2:
            pos_limit_len = np.random.normal(np.array([0.25, 0.75]) * world_length, 50).astype(int)
            pos_limit_width = np.random.normal(np.array([0.5, 0.5]) * world_length, 100).astype(int)

        elif number_of_objects == 3:
            pos_limit_len = np.random.normal(np.array([0.1, 0.5, 0.9]) * world_length, 50).astype(int)
            pos_limit_width = np.random.normal(np.array([0.1, 0.5, 0.9]) * world_length, 50).astype(int)

            if np.random.random() > 0.5:
                pos_limit_len = -1 * pos_limit_len

            if np.random.random() > 0.5:
                pos_limit_width = -1 * pos_limit_width

                if random.random() > 0.5:
                    pos_limit_width = 0 * pos_limit_width

        elif number_of_objects == 4:
            pos_limit_len = np.random.normal(np.array([0.25, 0.25, 0.75, 0.75]) * world_length, 50).astype(int)
            pos_limit_width = np.random.normal(np.array([0.25, 0.75, 0.25, 0.75]) * world_length, 50).astype(int)

        iterator = 0

        for world_object in objects:
            length = np.random.randint(150, 250) * 2 + 1
            width = np.random.randint(150, 250) * 2 + 1

            height = 0
            steps = random.randint(3, 8)
            symmetric = random.randint(0, 1)
            smoothing = (np.random.random(1) + 1) * 7.5
            period = (np.random.random(1) + 1) * np.pi / 2

            position = [0, 0, 0]
            position[0] = (random.random() - 0.5) * pos_limit_len[iterator]
            position[1] = (random.random() - 0.5) * pos_limit_width[iterator]
            position[2] = 0

            orientation = (np.random.random(3) - 0.5) * np.pi / 2
            orientation[2] *= 2

            if world_object == 'plank':
                height = random.random() * 0.5
                orientation[0] = random.random() * np.pi / 6
                orientation[1] = (random.random() - 0.5) * np.pi / 4
                orientation[2] = (random.random() - 0.5) * np.pi / 4

                if orientation[1] * orientation[2] > 0:
                    orientation[2] *= -1

            elif world_object == 'stairs':
                height = (random.random() + 3) / 5
                orientation[2] = (random.random() - 0.5) * np.pi / 2

            elif world_object == 'uneven_terrain':
                height = (random.random() + 1) * 0.025
                smoothing = random.random() * 4 + 6

            elif world_object == 'sine_wave_terrain':
                height = (random.random() + 1) * 0.05
                period = (random.random() + 0.5) * np.pi

            elif world_object == 'brick_terrain':
                height = random.random() * 0.06 + 0.04

            world.create_object(world_object, length, width, height, position, orientation, steps=steps,
                                symmetric=symmetric, smoothing=smoothing, period=period)

            iterator += 1

        height_map = world.get_height_map()

        save_path = None

        if save_as_png:
            save_path = HEIGHT_MAPS_PATH + CURRENT_DATETIME_STR + '/' + file_name_prefix + '_' + \
                '{0:0=3d}'.format(scene) + '.png'

        show_height_map(height_map, visualize_height_map=visualize, save_height_map_path=save_path, color_map='gray',
                        save_for_material=False)


if __name__ == '__main__':
    main()
