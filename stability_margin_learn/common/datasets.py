import os
import numpy as np

from scipy.spatial.transform import Rotation as R

from common.parser import DataParser
from common.paths import ProjectPaths


class TrainingDataset:
    def __init__(self, data_folder='stability_margin'):
        self._data_folder = data_folder
        self._data_parser = None

        self._data_offset = np.concatenate([
            np.array([0, 0, 1]),  # Rotation along gravity axis
            np.zeros(3),  # Linear velocity
            np.zeros(3),  # Linear acceleration
            np.zeros(3),  # Angular acceleration
            np.zeros(3),  # External force
            np.zeros(3),  # External torque
            np.array([0.36, 0.21, -0.47]),  # LF foot
            np.array([0.36, -0.21, -0.47]),  # RF foot
            np.array([-0.36, 0.21, -0.47]),  # LH foot
            np.array([-0.36, -0.21, -0.47]),  # RH foot
            np.array([0.5]),  # Friction
            np.ones(4) * 0.5,  # Feet in contact
            np.array([0, 0, 1] * 4),  # Contact normals
            np.zeros(1)  # Stability margin
        ])

        self._data_multiplier = np.concatenate([
            np.array([6.5, 5.5, 30]),  # Rotation along gravity axis
            np.array([5.0, 5.0, 20]),  # Linear velocity
            np.array([3.4, 3.4, 10]),  # Linear acceleration
            np.array([5.0, 5.0, 3.4]),  # Angular acceleration
            np.array([0.05, 0.05, 0.2]),  # External force
            np.array([0.05, 0.05, 0.2]),  # External torque
            np.ones(3) * 21.0,  # LF foot
            np.ones(3) * 21.0,  # RF foot
            np.ones(3) * 21.0,  # LH foot
            np.ones(3) * 21.0,  # RH foot
            np.ones(1) * 5.0,  # Friction
            np.ones(4) * 2.0,  # Feet in contact
            np.array([5.0, 5.0, 25] * 4),  # Contact normals
            np.ones(1) * 10.0  # Stability margin
        ])

    def get_data_offset(self):
        return self._data_offset

    def get_data_multiplier(self):
        return self._data_multiplier

    def get_data_folder(self):
        return self._data_folder

    def get_training_data_parser(self, process_data=True, max_files=None):
        if self._data_parser is None:
            # Get training data
            data_parser = DataParser()
            paths = ProjectPaths()

            num_files = 0

            for path, _, files in os.walk(paths.DATA_PATH + '/' + self._data_folder):
                for file_name in files:
                    if max_files is not None and num_files >= max_files:
                        break

                    if file_name.endswith('.csv'):
                        data_parser.append(filename=os.path.join(path, file_name))

                        num_files += 1
                        print '\rFiles Processed: {}'.format(num_files),
                if max_files is not None and num_files >= max_files:
                    break
            print

            # Update the data in the data parser object and split it as input and output
            training_data = data_parser.data()

            rotation_matrices = R.from_euler('XYZ', training_data[:, :3], degrees=False).as_dcm()
            rotation_matrices_inv = np.transpose(rotation_matrices, (0, 2, 1))

            # Convert to the base frame
            training_data[:, 18:21] = np.einsum('aik,ak->ai', rotation_matrices_inv,
                                                training_data[:, 18:21])  # LF Foot Position
            training_data[:, 21:24] = np.einsum('aik,ak->ai', rotation_matrices_inv,
                                                training_data[:, 21:24])  # RF Foot Position
            training_data[:, 24:27] = np.einsum('aik,ak->ai', rotation_matrices_inv,
                                                training_data[:, 24:27])  # LH Foot Position
            training_data[:, 27:30] = np.einsum('aik,ak->ai', rotation_matrices_inv,
                                                training_data[:, 27:30])  # RH Foot Position

            # If the foot is not in stance, change its position to nominal
            training_data[np.argwhere(training_data[:, 31] == 0), 18:21] = np.array([0.36, 0.21, -0.47])
            training_data[np.argwhere(training_data[:, 32] == 0), 21:24] = np.array([0.36, -0.21, -0.47])
            training_data[np.argwhere(training_data[:, 33] == 0), 24:27] = np.array([-0.36, 0.21, -0.47])
            training_data[np.argwhere(training_data[:, 34] == 0), 27:30] = np.array([-0.36, -0.21, -0.47])

            # If the foot is not in stance, set its contact normal to vertical
            training_data[np.argwhere(training_data[:, 31] == 0), 35:38] = np.array([0.0, 0.0, 1.0])
            training_data[np.argwhere(training_data[:, 32] == 0), 38:41] = np.array([0.0, 0.0, 1.0])
            training_data[np.argwhere(training_data[:, 33] == 0), 41:44] = np.array([0.0, 0.0, 1.0])
            training_data[np.argwhere(training_data[:, 34] == 0), 44:47] = np.array([0.0, 0.0, 1.0])

            training_data = np.hstack([
                rotation_matrices[:, 2],  # Rotation along gravity axis
                np.einsum('aik,ak->ai', rotation_matrices_inv, training_data[:, 3:6]),  # Linear Velocity
                np.einsum('aik,ak->ai', rotation_matrices_inv, training_data[:, 6:9]),  # Linear Acceleration
                np.einsum('aik,ak->ai', rotation_matrices_inv, training_data[:, 9:12]),  # Angular Acceleration
                np.einsum('aik,ak->ai', rotation_matrices_inv, training_data[:, 12:15]),  # Ext Force in base frame
                np.einsum('aik,ak->ai', rotation_matrices_inv, training_data[:, 15:18]),  # Ext Torque in base frame
                training_data[:, 18:21],  # LF Foot Position: base frame
                training_data[:, 21:24],  # LH Foot Position: base frame
                training_data[:, 24:27],  # RF Foot Position: base frame
                training_data[:, 27:30],  # RH Foot Position: base frame
                training_data[:, 30].reshape(-1, 1),  # Friction
                training_data[:, 31:35],  # Feet in contact
                training_data[:, 35:47],  # Contact normals
                training_data[:, 47].reshape(-1, 1)  # Stability Margin
            ])

            if process_data:
                training_data = (training_data - self._data_offset) * self._data_multiplier

            data_parser.update_data(training_data)
            data_parser.set_io_split_index(-1)

            # Divide the dataset into training, validation and test
            data_parser.divide_dataset()

            self._data_parser = data_parser
            return data_parser
        else:
            return self._data_parser

    def scale_data(self, data, input_only=False):
        if input_only:
            return (data - self._data_offset[:-1]) * self._data_multiplier[:-1]

        return (data - self._data_offset) * self._data_multiplier
