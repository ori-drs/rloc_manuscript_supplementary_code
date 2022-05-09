import numpy as np
import pandas as pd

import torch
import torch.utils.data as utils


class DataParser:
    def __init__(self, filename=None, delimiter=',', dtype=np.float):
        self._filename = filename
        self._delimiter = delimiter
        self._dtype = dtype

        if self._filename is not None:
            self._data = self._load_data()
        else:
            self._data = None

        self._training_data = None
        self._validation_data = None
        self._test_data = None

        self._torch_dataset = None
        self._torch_dataloader = None

        self._split_index = -1

    def _load_data(self, filename=None, append=False):
        if filename is None and self._filename is not None:
            filename = self._filename

        elif filename is None and self._filename is None:
            return None

        data = pd.read_csv(filename, delimiter=self._delimiter, skipinitialspace=True,
                           dtype=self._dtype, header=None).to_numpy()
        data = data[~np.isnan(data).any(axis=1)]

        if data.shape[1] == 48:
            if append:
                if self._data is not None:
                    self._data = np.append(self._data, data, axis=0)
                else:
                    self._data = data
        else:
            print "file is empty ", filename, "skip append"

        return data

    def append(self, filename):
        self._load_data(filename, append=True)

    def data(self):
        return self._data

    def data_shape(self):
        if self._data is not None:
            return self._data.shape
        else:
            return None

    def update_data(self, data, shuffle=True):
        if shuffle:
            np.random.shuffle(data)

        self._data = data

    def divide_dataset(self, training=0.7, validation=0.2):
        assert training + validation < 1

        training_data_last_index = int(training * self._data.shape[0])
        validation_data_last_index = training_data_last_index + int(validation * self._data.shape[0])

        self._training_data = self._data[:training_data_last_index, :]
        self._validation_data = self._data[training_data_last_index:validation_data_last_index, :]
        self._test_data = self._data[validation_data_last_index:, :]

    def torch_dataset(self, data_type=None):
        if data_type == 'training':
            tensor_input = torch.Tensor(self._training_data[:, :self._split_index])
            tensor_target = torch.Tensor(self._training_data[:, self._split_index:]).reshape(
                (self._training_data.shape[0], -1)
            )

        elif data_type == 'validation':
            tensor_input = torch.Tensor(self._validation_data[:, :self._split_index])
            tensor_target = torch.Tensor(self._validation_data[:, self._split_index:]).reshape(
                (self._validation_data.shape[0], -1)
            )

        elif data_type == 'testing':
            tensor_input = torch.Tensor(self._test_data[:, :self._split_index])
            tensor_target = torch.Tensor(self._test_data[:, self._split_index:]).reshape((self._test_data.shape[0], -1))

        else:
            tensor_input = torch.Tensor(self._data[:, :self._split_index])
            tensor_target = torch.Tensor(self._data[:, self._split_index:]).reshape((self._data.shape[0], -1))

        self._torch_dataset = utils.TensorDataset(tensor_input, tensor_target)

        return self._torch_dataset

    def torch_dataloader(self, params, data_type=None):
        self._torch_dataloader = utils.DataLoader(self.torch_dataset(data_type=data_type), **params)
        return self._torch_dataloader

    def get_inputs(self, data_type=None):
        if data_type == 'training':
            return self._training_data[:, :self._split_index]

        elif data_type == 'validation':
            return self._validation_data[:, :self._split_index]

        elif data_type == 'testing':
            return self._test_data[:, :self._split_index]

        else:
            return self._data[:, :self._split_index]

    def get_targets(self, data_type=None):
        if data_type == 'training':
            return self._training_data[:, self._split_index:]

        elif data_type == 'validation':
            return self._validation_data[:, self._split_index:]

        elif data_type == 'testing':
            return self._test_data[:, self._split_index:]

        else:
            return self._data[:, self._split_index:]

    def set_io_split_index(self, index):
        assert index < self._data.shape[1]
        self._split_index = index

    def clear_data(self):
        self._data = None

    def save_data(self, save_path, samples=None, start_index=0):
        if samples is not None:
            assert type(samples) == int

            if start_index == 'random':
                start_index = np.random.randint(self._data.shape[0] - samples - 1)
            else:
                assert type(start_index) == int

            np.savetxt(save_path, self._data[start_index: start_index + samples, :], delimiter=self._delimiter,
                       newline='\n', fmt='%1.10f')

        else:
            assert type(start_index) == int
            np.savetxt(save_path, self._data[start_index:, :], delimiter=self._delimiter, newline='\n', fmt='%1.10f')
