import imageio
import numpy as np

import os
from common.paths import HEIGHT_MAPS_PATH
from common.visualizer import show_height_map

height_scaling = 1

visualize_height_map = True
save_height_map_path = None


def main():
    height_map_filename = 'scene_test.png'
    height_map_path = HEIGHT_MAPS_PATH + height_map_filename

    height_map = np.array(imageio.imread(height_map_path)) * height_scaling

    print('\nImage Path: ', os.path.abspath(height_map_path))
    print('Image Shape: ', np.shape(height_map))
    print('Maximum Value: ', np.max(height_map))
    print('Minimum Value: ', np.min(height_map))
    print('Mean Value: ', np.mean(height_map))
    print('Standard Deviation: ', np.std(height_map))

    show_height_map(height_map=height_map_path, visualize_height_map=visualize_height_map,
                    save_height_map_path=save_height_map_path, color_map='gray')


if __name__ == '__main__':
    main()
