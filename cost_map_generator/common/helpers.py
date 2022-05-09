import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import imageio
import math
import png


class HeightMapHelper:
    def __init__(self):
        self.smoothing_kernel = np.array([[1, 1, 1],
                                          [1, 2, 1],
                                          [1, 1, 1]])
        self.smoothing_kernel = self.smoothing_kernel / self.smoothing_kernel.sum()

        self.laplacian_kernel = np.array([[-1, -1, -1],
                                          [-1, 8., -1],
                                          [-1, -1, -1]])

        self.blur_kernel = np.ones((3, 3)) / 9

    def apply_smoothing_kernel(self, source):
        return self.convolution(source, self.smoothing_kernel)

    def apply_laplacian_kernel(self, source):
        return self.convolution(source, self.laplacian_kernel)

    def apply_blur_kernel(self, source):
        return self.convolution(source, self.blur_kernel)

    def get_cost_map(self, source, filter_size=9):
        product_filter = np.zeros((filter_size + 2, filter_size + 2))
        radial_len = math.floor((filter_size + 1) / 2)

        for row in range(product_filter.shape[0]):
            for col in range(product_filter.shape[1]):
                radius = ((row - radial_len) ** 2 + (col - radial_len) ** 2) ** 0.5
                product_filter[row, col] = 0.0 if (radius >= radial_len) else (radial_len - radius) / radial_len
        return self.convolution(source, product_filter[1:-1, 1:-1])

    @staticmethod
    def convolution(source, kernel):
        filtered_matrix = np.zeros((source.shape[0] - kernel.shape[0] + 1, source.shape[1] - kernel.shape[1] + 1))

        for row in range(filtered_matrix.shape[0]):
            for col in range(filtered_matrix.shape[1]):
                for k_row in range(kernel.shape[0]):
                    for k_col in range(kernel.shape[1]):
                        filtered_matrix[row, col] += kernel[k_row, k_col] * source[row + k_row][col + k_col]

        return filtered_matrix

    @staticmethod
    def get_height_map_from_path(height_map_path):
        return np.array(imageio.imread(height_map_path))

    @staticmethod
    def show_height_map(height_map, process_height_map=True, visualize_height_map=True, save_height_map_path=None,
                        save_for_material=False, color_map='PuBuGn'):
        if visualize_height_map or save_height_map_path is not None:
            if isinstance(height_map, str):
                height_map_filename = height_map.split('/')[-1]
                height_map = np.array(imageio.imread(height_map))
            elif save_height_map_path is not None:
                height_map_filename = save_height_map_path.split('/')[-1]
            else:
                height_map_filename = ''

            mpl.rcParams['toolbar'] = 'None'

            if process_height_map:
                height_map = np.clip(height_map, -1, 1)
                height_map = 65535 * (height_map + 1) / 2

            plt.imshow(height_map, aspect='auto', interpolation='none', cmap=color_map)
            plt.xlim(0, np.shape(height_map)[1] - 1)
            plt.ylim(np.shape(height_map)[0] - 1, 0)

            fig = plt.gcf()
            fig.canvas.set_window_title(height_map_filename)
            plt.tight_layout()

            plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                                hspace=0, wspace=0)
            plt.margins(0, 0)
            plt.axis('off')
            plt.rcParams["xtick.direction"] = "in"
            plt.rcParams["ytick.direction"] = "in"

            if save_height_map_path is not None:
                if save_for_material:
                    plt.savefig(save_height_map_path)
                else:
                    writer = png.Writer(width=height_map.shape[1], height=height_map.shape[0], bitdepth=16,
                                        colormap=True)
                    with open(save_height_map_path, 'wb') as f:
                        writer.write(f, height_map.astype(int).tolist())

            if visualize_height_map:
                plt.show()
