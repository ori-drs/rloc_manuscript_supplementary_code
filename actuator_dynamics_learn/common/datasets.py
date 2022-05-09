import os
import numpy as np

from common.parser import DataParser
from common.paths import ProjectPaths


class TrainingDataset:
    def __init__(self, data_folder='anymal'):
        self._data_folder = data_folder

    def get_data_folder(self):
        return self._data_folder

    def get_training_data_parser(self, process_data=True):
        # Get training data
        data_parser = DataParser()
        paths = ProjectPaths()

        actuator_training_data_list = []

        for path, _, files in os.walk(paths.DATA_PATH + '/' + self._data_folder):
            for file_name in files:
                if file_name.endswith('.csv'):
                    data_parser.clear_data()
                    data_parser.append(filename=os.path.join(path, file_name))
                    data = data_parser.data()

                    if process_data:
                        # Network input data holder
                        network_input_data_singular = np.array([
                            data[:, 3] - data[:, 0],  # Joint Position Error (Command - Measured)
                            data[:, 4] - data[:, 1],  # Joint Velocity Error
                            data[:, 5] - data[:, 2],  # Joint Torque Error
                            data[:, 4],  # Joint Velocity Command
                            data[:, 5],  # Joint Torque Command
                            data[:, 6],  # Joint P Gain
                            data[:, 7]  # Joint D Gain
                        ]).T

                        network_output_data = np.array([data[:, 2]]).T

                        row_start = 8
                        row_end = data.shape[0] - 1

                        training_data = np.zeros((row_end - row_start, 22))

                        # Network training data which includes history
                        for row in range(row_start, row_end):
                            # Joint Position Error History
                            training_data[row - row_start, 0] = network_input_data_singular[row - row_start, 0]
                            training_data[row - row_start, 1] = network_input_data_singular[row - int(row_start / 2), 0]
                            training_data[row - row_start, 2] = network_input_data_singular[row, 0]

                            # Joint Velocity Error History
                            training_data[row - row_start, 3] = network_input_data_singular[row - row_start, 1]
                            training_data[row - row_start, 4] = network_input_data_singular[row - int(row_start / 2), 1]
                            training_data[row - row_start, 5] = network_input_data_singular[row, 1]

                            # Joint Torque Measured History
                            training_data[row - row_start, 6] = network_input_data_singular[row - row_start, 2]
                            training_data[row - row_start, 7] = network_input_data_singular[row - int(row_start / 2), 2]
                            training_data[row - row_start, 8] = network_input_data_singular[row, 2]

                            # Joint Velocity Command History
                            training_data[row - row_start, 9] = network_input_data_singular[row - row_start, 3]
                            training_data[row - row_start, 10] = network_input_data_singular[
                                row - int(row_start / 2), 3]
                            training_data[row - row_start, 11] = network_input_data_singular[row, 3]

                            # Joint Feed-forward Torque Command History
                            training_data[row - row_start, 12] = network_input_data_singular[row - row_start, 4]
                            training_data[row - row_start, 13] = network_input_data_singular[
                                row - int(row_start / 2), 4]
                            training_data[row - row_start, 14] = network_input_data_singular[row, 4]

                            # Joint P Gain History
                            training_data[row - row_start, 15] = network_input_data_singular[row - row_start, 5]
                            training_data[row - row_start, 16] = network_input_data_singular[
                                row - int(row_start / 2), 5]
                            training_data[row - row_start, 17] = network_input_data_singular[row, 5]

                            # Joint D Gain History
                            training_data[row - row_start, 18] = network_input_data_singular[row - row_start, 6]
                            training_data[row - row_start, 19] = network_input_data_singular[
                                row - int(row_start / 2), 6]
                            training_data[row - row_start, 20] = network_input_data_singular[row, 6]

                            training_data[row - row_start, 21] = network_output_data[row + 1]
                    else:
                        training_data = data_parser.data()

                    actuator_training_data_list.append(training_data)

        # Update the data in the data parser object and split it as input and output
        training_data = np.concatenate(actuator_training_data_list)

        if process_data:
            training_data_offset = np.concatenate([
                np.zeros(3),  # Joint Position Error
                np.zeros(3),  # Joint Velocity Error
                np.zeros(3),  # Joint Torque Measured
                np.zeros(3),  # Joint Velocity Command
                np.zeros(3),  # Joint Feed-Forward Torque,
                np.ones(3) * 100,  # Joint P Gain
                np.ones(3) * 0.5,  # Joint D Gain
                np.zeros(1)  # Output Torque
            ])

            training_data_multiplier = np.concatenate([
                np.ones(3) * 0.5,  # Joint Position Error
                np.ones(3) * 0.0625,  # Joint Velocity Error
                np.ones(3) / 42.0,  # Joint Torque Measured
                np.ones(3) * 0.0625,  # Joint Velocity Command
                np.ones(3) / 42.0,  # Joint Feed-Forward Torque
                np.ones(3) * 0.013,  # Joint P Gain
                np.ones(3) * 2.0,  # Joint D Gain
                np.ones(1) / 42.0  # Output Torque
            ])

            training_data = (training_data - training_data_offset) * training_data_multiplier

        data_parser.update_data(training_data)
        data_parser.set_io_split_index(-1)

        # Divide the dataset into training, validation and test
        data_parser.divide_dataset()

        return data_parser
