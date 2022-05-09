# Note: This script is only to demonstrate the generation of cost map.
# Please refer to the manuscript for actual description of cost map generation.

import os
import numpy as np

from common.paths import ProjectPaths
from common.helpers import HeightMapHelper

# INDEX_START = 0: For computing the cost map for the entire terrain.
# INDEX_START = 385: For computing the cost map of a cropped terrain.
# INDEX_START = 490: For computing local cost map.
INDEX_START = 385


def main():
    paths = ProjectPaths()
    height_map_helper = HeightMapHelper()

    # Get test image path
    height_map_filename = 'scene_bricks.png'
    height_map_path = paths.DATA_PATH + '/height_maps/' + height_map_filename

    # Import height map as array from image path
    height_map = height_map_helper.get_height_map_from_path(height_map_path)

    if INDEX_START > 0:
        lim = INDEX_START
        height_map = height_map[lim:-lim, lim:-lim]

    print(height_map.shape)

    # Directory to save the processed height maps
    save_height_map_dir = paths.RESULTS_PATH + '/cost_maps/' + paths.INIT_DATETIME_STR
    os.makedirs(save_height_map_dir, exist_ok=True)

    # Show the original height map
    height_map_helper.show_height_map(height_map=height_map, process_height_map=False,
                                      visualize_height_map=True, save_for_material=True,
                                      save_height_map_path=save_height_map_dir + '/original.png')

    # Smoothen the height map and show
    smooth_map = height_map_helper.apply_smoothing_kernel(height_map)
    height_map_helper.show_height_map(height_map=smooth_map, process_height_map=False,
                                      visualize_height_map=True, save_for_material=True,
                                      save_height_map_path=save_height_map_dir + '/smooth.png')

    # Perform laplacian on the smoothened kernel and show
    slope_derivative_map = height_map_helper.apply_laplacian_kernel(smooth_map)
    height_map_helper.show_height_map(height_map=slope_derivative_map, process_height_map=False,
                                      visualize_height_map=True, save_for_material=True,
                                      save_height_map_path=save_height_map_dir + '/laplacian.png')

    # Absolute of laplacian
    abs_slope_derivative_map = np.abs(slope_derivative_map)
    height_map_helper.show_height_map(height_map=abs_slope_derivative_map, process_height_map=False,
                                      visualize_height_map=True, save_for_material=True,
                                      save_height_map_path=save_height_map_dir + '/abs_slope_derivative_map.png')

    # Apply blurring filter
    abs_blurred_map = height_map_helper.apply_blur_kernel(abs_slope_derivative_map)
    height_map_helper.show_height_map(height_map=abs_blurred_map, process_height_map=False,
                                      visualize_height_map=True, save_for_material=True,
                                      save_height_map_path=save_height_map_dir + '/abs_blurred_map.png')

    # Introduce a filter which only uses a circular image and also reduces the intensity from the center
    # This is only done for the elevation map local to a foot and not for the entire terrain. We only
    # include this here to demonstrate the circular filtering process.
    product_filter = np.zeros(abs_blurred_map.shape)

    radial_length = min(product_filter.shape[0], product_filter.shape[1]) / 2

    for row in range(product_filter.shape[0]):
        for col in range(product_filter.shape[1]):
            radius = ((row - radial_length) ** 2 + (col - radial_length) ** 2) ** 0.5
            product_filter[row, col] = 0.0 if (radius >= radial_length) else (radial_length - radius) / radial_length

    filtered_derivative_map = np.multiply(abs_blurred_map, product_filter)
    height_map_helper.show_height_map(height_map=filtered_derivative_map, process_height_map=False,
                                      visualize_height_map=True, save_for_material=True,
                                      save_height_map_path=save_height_map_dir + '/circular_filtered_derivative.png')

    print(filtered_derivative_map.shape)

    # Get the cost map (for the whole terrain) - the circular distance filter here is applied locally along each CELL.
    # Unlike in the previous case pf `product_filter` where the circular distance filter is used as a mask, for the
    # cost map generated here, we perform convolutions using the circular distance filter. This results in the 
    # entire terrain cost map being generated at once.
    cost_map = height_map_helper.get_cost_map(abs_blurred_map)
    height_map_helper.show_height_map(height_map=cost_map, process_height_map=False,
                                      visualize_height_map=True, save_for_material=True,
                                      save_height_map_path=save_height_map_dir + '/terrain_cost_map.png')


if __name__ == '__main__':
    main()
