import numpy as np
import math

from scipy.spatial.transform import Rotation as rot
from scipy.ndimage.filters import median_filter, gaussian_filter


class World:
    def __init__(self, length, width, height=0):
        self.length = length
        self.width = width
        self.height = height

        self._height_map = np.zeros((self.length, self.width))

    def create_object(self, object_type, length, width, breadth, position, orientation, steps=1, symmetric=False,
                      smoothing=1, period=np.pi / 2):
        if object_type == 'plank':
            self._height_map = Plank(self._height_map, length, width, breadth, position, orientation).world_height_map

        elif object_type == 'stairs':
            self._height_map = Stairs(self._height_map, length, width, breadth, position, orientation, steps,
                                      symmetric).world_height_map

        elif object_type == 'uneven_terrain':
            self._height_map = UnevenTerrain(self._height_map, length, width, breadth, position, orientation,
                                             smoothing).world_height_map

        elif object_type == 'sine_wave_terrain':
            self._height_map = SineWaveTerrain(self._height_map, length, width, breadth, position, orientation,
                                               period).world_height_map

        elif object_type == 'brick_terrain':
            self._height_map = BrickTerrain(self._height_map, length, breadth, position,
                                            orientation).world_height_map

    def get_height_map(self):
        return np.around(self._height_map, decimals=4) + self.height

    def reset_world(self, length=None, width=None):
        if length is None:
            length = self.length

        if width is None:
            width = self.width

        self._height_map = np.zeros((length, width))


class Plank:
    def __init__(self, world_height_map, length, width, breadth, position, orientation):
        self.world_height_map = world_height_map
        self._world_height = np.min(self.world_height_map)

        self._length = width
        self._width = length
        self._thickness = breadth
        self._translation = position

        # Create an (n, 3) matrix to get (x, y, h[x,y]) where n = length * width
        self._plank_grid_map = np.array(np.meshgrid(
            np.arange(self._length) - math.floor((self._length - 1) / 2),
            np.arange(self._width) - math.floor((self._width - 1) / 2),
        )).reshape(2, -1)

        self._plank_grid_map = np.array([
            self._plank_grid_map[0],
            self._plank_grid_map[1],
            np.ones(np.shape(self._plank_grid_map[0]))
        ])

        self._plank_grid_map = np.transpose(self._plank_grid_map)

        # Apply rotation
        orientation[2] = orientation[2] + np.pi
        self._rotation = rot.from_euler('xyz', orientation)
        self._plank_grid_map = self._rotation.apply(self._plank_grid_map).T

        # Perform interpolation
        x_coordinates = self._plank_grid_map[0]
        y_coordinates = self._plank_grid_map[1]

        heights = self._plank_grid_map[2]
        heights = heights * self._thickness / np.max(heights)

        grid_x_coordinates = np.ravel(np.fix(x_coordinates))
        grid_y_coordinates = np.ravel(np.fix(y_coordinates))

        # Update World Matrix
        for iterator in range(np.shape(grid_x_coordinates.astype(int))[0]):
            try:
                row = grid_x_coordinates.astype(int)[iterator] + \
                      np.floor((np.shape(self.world_height_map)[0] - 1) / 2).astype(int) + \
                      np.around(self._translation[0]).astype(int)
                col = grid_y_coordinates.astype(int)[iterator] + \
                      np.floor((np.shape(self.world_height_map)[1] - 1) / 2).astype(int) + \
                      np.around(self._translation[1]).astype(int)

                if row >= 0 and col >= 0:
                    self.world_height_map[row, col] = heights[iterator]

            except IndexError:
                pass

        self.world_height_map = median_filter(self.world_height_map, 4)


class Stairs:
    def __init__(self, world_height_map, length, width, breadth, position, orientation, steps=1, symmetric=False):
        self.world_height_map = world_height_map
        self._length = width
        self._width = length
        self._height = breadth
        self._steps = (np.fix(steps / 2) * 2 + 1).astype(int)
        self._translation = position
        self._yaw = orientation[2] + np.pi

        self._step_length = int(np.fix(self._length / (2 * self._steps)) * 2 + 1)
        self._step_width = np.fix(self._width / 2) * 2 + 1
        self._step_height = self._height / self._steps

        self._length = int(self._step_length * self._steps)

        self._stairs_height_map = np.ones((self._length, self._width))

        if not symmetric:
            for step in range(self._steps):
                self._stairs_height_map[step * self._step_length:(steps + 1) * self._step_length, :] = \
                    self._step_height * (step + 1)
        else:
            for step in range(int(np.fix(self._steps / 2) + 1)):
                self._stairs_height_map[step * self._step_length:(steps + 1) * self._step_length, :] = \
                    self._step_height * (step + 1)
            for step in range(int(np.fix(self._steps / 2) + 1), self._steps):
                self._stairs_height_map[step * self._step_length:(steps + 1) * self._step_length, :] = \
                    self._step_height * (self._steps - step)

        self._stairs_grid_map = np.array(np.meshgrid(
            np.arange(self._length) - math.floor((self._length - 1) / 2),
            np.arange(self._width) - math.floor((self._width - 1) / 2),
        )).reshape(2, -1)

        self._stairs_grid_map = np.array([
            self._stairs_grid_map[0],
            self._stairs_grid_map[1],
            self._stairs_height_map.flatten()
        ])

        self._rotation = rot.from_euler('z', self._yaw)
        self._stairs_grid_map = self._rotation.apply(self._stairs_grid_map.T).T

        x_coordinates = self._stairs_grid_map[0]
        y_coordinates = self._stairs_grid_map[1]

        heights = self._stairs_grid_map[2]

        grid_x_coordinates = np.ravel(np.fix(x_coordinates))
        grid_y_coordinates = np.ravel(np.fix(y_coordinates))

        # Update World Matrix
        for iterator in range(np.shape(grid_x_coordinates.astype(int))[0]):
            try:
                row = grid_x_coordinates.astype(int)[iterator] + \
                      np.floor((np.shape(self.world_height_map)[0] - 1) / 2).astype(int) + \
                      np.around(self._translation[0]).astype(int)
                col = grid_y_coordinates.astype(int)[iterator] + \
                      np.floor((np.shape(self.world_height_map)[1] - 1) / 2).astype(int) + \
                      np.around(self._translation[1]).astype(int)

                if row >= 0 and col >= 0:
                    self.world_height_map[row, col] = heights[iterator]
            except IndexError:
                pass

        self.world_height_map = median_filter(self.world_height_map, 2)


class UnevenTerrain:
    def __init__(self, world_height_map, length, width, breadth, position, orientation, smoothing):
        self.world_height_map = world_height_map
        self._length = int(np.fix(length / 2) * 2 + 1)
        self._width = int(np.fix(width / 2) * 2 + 1)
        self._breadth = breadth
        self._smoothing = max(0, smoothing)

        self._translation = position
        self._orientation = orientation

        self._smoothing = smoothing

        self._uneven_height_map = (np.random.rand(self._length, self._width) - 0.5) * 2 * self._breadth
        self._uneven_height_map = gaussian_filter(self._uneven_height_map, int(5 * np.tanh(self._smoothing / 10)))
        self._uneven_height_map = self._breadth * self._uneven_height_map / np.max(self._uneven_height_map)

        # if self._length * self._width < 120000:
        #     self._uneven_grid_map = np.array(np.meshgrid(
        #         np.arange(self._length) - math.floor((self._length - 1) / 2),
        #         np.arange(self._width) - math.floor((self._width - 1) / 2),
        #     )).reshape(2, -1)
        #
        #     self._uneven_grid_map = np.array([
        #         self._uneven_grid_map[0],
        #         self._uneven_grid_map[1],
        #         self._uneven_height_map.flatten()
        #     ])
        #
        #     self._rotation = rot.from_euler('xyz', self._orientation)
        #     self._uneven_grid_map = self._rotation.apply(self._uneven_grid_map.T).T
        #
        #     x_coordinates = self._uneven_grid_map[0]
        #     y_coordinates = self._uneven_grid_map[1]
        #
        #     heights = self._uneven_grid_map[2]
        #
        #     grid_x_coordinates = np.ravel(np.fix(x_coordinates))
        #     grid_y_coordinates = np.ravel(np.fix(y_coordinates))
        #
        #     # Update World Matrix
        #     for iterator in range(np.shape(grid_x_coordinates.astype(int))[0]):
        #         try:
        #             self.world_height_map[
        #                 grid_x_coordinates.astype(int)[iterator] +
        #                 np.floor((np.shape(self.world_height_map)[0] - 1) / 2).astype(int) +
        #                 np.around(self._translation[0]).astype(int),
        #                 grid_y_coordinates.astype(int)[iterator] +
        #                 np.floor((np.shape(self.world_height_map)[1] - 1) / 2).astype(int) +
        #                 np.around(self._translation[1]).astype(int)
        #             ] = heights[iterator]
        #         except IndexError:
        #             pass
        #
        # else:

        for x in range(self._length):
            for y in range(self._width):
                try:
                    row = x + np.floor((np.shape(self.world_height_map)[0] - 1) / 2).astype(int) - np.floor(
                        (self._length - 1) / 2).astype(int)
                    col = y + np.floor((np.shape(self.world_height_map)[1] - 1) / 2).astype(int) - np.floor(
                        (self._width - 1) / 2).astype(int)

                    if row >= 0 and col >= 0:
                        self.world_height_map[row, col] = self._uneven_height_map[x, y]

                except IndexError:
                    pass


class SineWaveTerrain:
    def __init__(self, world_height_map, length, width, breadth, position, orientation, period):
        self.world_height_map = world_height_map

        self._width = int(np.fix(length / 2) * 2 + 1)
        self._length = int(np.fix(width / 2) * 2 + 1)
        self._amplitude = breadth
        self._translation = position
        self._yaw = orientation[2] + np.pi
        self._period = period

        self._sine_height_map = np.zeros((self._length, self._width))

        for t in range(self._length):
            self._sine_height_map[t, :] = self._amplitude * np.sin(0.05 * t * self._period)

        self._sine_grid_map = np.array(np.meshgrid(
            np.arange(self._length) - math.floor((self._length - 1) / 2),
            np.arange(self._width) - math.floor((self._width - 1) / 2),
        )).reshape(2, -1)

        self._sine_grid_map = np.array([
            self._sine_grid_map[0],
            self._sine_grid_map[1],
            self._sine_height_map.flatten()
        ])

        if self._yaw != 0:
            self._rotation = rot.from_euler('z', self._yaw)
            self._sine_grid_map = self._rotation.apply(self._sine_grid_map.T).T

        x_coordinates = self._sine_grid_map[0]
        y_coordinates = self._sine_grid_map[1]

        heights = self._sine_grid_map[2]

        grid_x_coordinates = np.ravel(np.fix(x_coordinates))
        grid_y_coordinates = np.ravel(np.fix(y_coordinates))

        # Update World Matrix
        for iterator in range(np.shape(grid_x_coordinates.astype(int))[0]):
            try:
                row = grid_x_coordinates.astype(int)[iterator] + \
                      np.floor((np.shape(self.world_height_map)[0] - 1) / 2).astype(int) + \
                      np.around(self._translation[0]).astype(int)
                col = grid_y_coordinates.astype(int)[iterator] + \
                      np.floor((np.shape(self.world_height_map)[1] - 1) / 2).astype(int) + \
                      np.around(self._translation[1]).astype(int)

                if row >= 0 and col >= 0:
                    self.world_height_map[row, col] = heights[iterator]

            except IndexError:
                pass

        self.world_height_map = gaussian_filter(self.world_height_map, 4)


class BrickTerrain:
    def __init__(self, world_height_map, length, breadth, position, orientation):
        self.world_height_map = world_height_map

        self._scale = 20

        self._length = int(max(200, length) / self._scale) * self._scale
        self._width = self._length
        self._breadth = breadth

        self._translation = position
        self._yaw = orientation[2]

        self._brick_height_map = np.random.randint(2, size=(
            int(self._length / self._scale), int(self._width / self._scale))) - np.random.randint(2, size=(
            int(self._length / self._scale), int(self._width / self._scale)))

        self._brick_height_map = self._brick_height_map.astype(float) * self._breadth

        self._brick_height_map = np.repeat(np.repeat(self._brick_height_map, self._scale, axis=0), self._scale, axis=1)

        self._brick_grid_map = np.array(np.meshgrid(
            np.arange(self._length) - math.floor((self._length - 1) / 2),
            np.arange(self._width) - math.floor((self._width - 1) / 2),
        )).reshape(2, -1)

        self._brick_grid_map = np.array([
            self._brick_grid_map[0],
            self._brick_grid_map[1],
            self._brick_height_map.flatten()
        ])

        self._rotation = rot.from_euler('z', self._yaw)
        self._brick_grid_map = self._rotation.apply(self._brick_grid_map.T).T

        x_coordinates = self._brick_grid_map[0]
        y_coordinates = self._brick_grid_map[1]

        heights = self._brick_grid_map[2]

        grid_x_coordinates = np.ravel(np.fix(x_coordinates))
        grid_y_coordinates = np.ravel(np.fix(y_coordinates))

        # Update World Matrix
        for iterator in range(np.shape(grid_x_coordinates.astype(int))[0]):
            try:
                row = grid_x_coordinates.astype(int)[iterator] + \
                      np.floor((np.shape(self.world_height_map)[0] - 1) / 2).astype(int) + \
                      np.around(self._translation[0]).astype(int)
                col = grid_y_coordinates.astype(int)[iterator] + \
                      np.floor((np.shape(self.world_height_map)[1] - 1) / 2).astype(int) + \
                      np.around(self._translation[1]).astype(int)

                if row >= 0 and col >= 0:
                    self.world_height_map[row, col] = heights[iterator]

            except IndexError:
                pass

        if self._yaw != 0:
            self.world_height_map = median_filter(self.world_height_map, 4)
