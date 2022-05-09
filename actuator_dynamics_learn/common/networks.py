import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class MultiLayerPerceptron(nn.Module):
    def __init__(self, in_dim=21, out_dim=1, hidden_layers=None, activation=F.softsign, dropout=0.2):
        super(MultiLayerPerceptron, self).__init__()

        # Network Parameters
        self._in_dim = in_dim
        self._out_dim = out_dim

        if hidden_layers is None:
            self._layers = [self._in_dim, 48, 48, self._out_dim]
        else:
            assert type(hidden_layers) == list

            self._layers = hidden_layers
            self._layers.insert(0, self._in_dim)
            self._layers.insert(len(self._layers), self._out_dim)

        self._num_layers = len(self._layers)
        self._activation = activation
        self._dropout = nn.Dropout(p=dropout)

        # Network Layers
        self._fully_connected_layers = nn.ModuleList(
            [nn.Linear(self._layers[layer], self._layers[layer + 1]) for layer in range(self._num_layers - 1)]
        )

    def forward(self, t, output_activation=None):
        for fully_connected_layer in self._fully_connected_layers[:-1]:
            t = self._dropout(self._activation(fully_connected_layer(t)))

        t = self._fully_connected_layers[-1](t)

        if output_activation is not None:
            if type(output_activation) == bool:
                if output_activation:
                    return self._activation(t)
                else:
                    return t
            else:
                return output_activation(t)

        return t

    def get_module_list(self):
        return self._fully_connected_layers

    def get_trainable_parameters(self):
        parameters = 0

        for params in list(self.parameters()):
            elements = 1

            for size in list(params.size()):
                elements *= size

            parameters += elements

        return parameters

    def set_dropout(self, p):
        self._dropout = nn.Dropout(p=min(max(0, p), 1))

    def load_params_from_txt(self, file_path):
        loaded_params = np.loadtxt(file_path, delimiter=', ')

        network_params = []
        iterator = 0

        network_state_dict = self.state_dict()
        device = self.get_device()

        for layer in range(self._num_layers - 1):
            network_params.append(
                (loaded_params[iterator: iterator + self._layers[layer] * self._layers[layer + 1]]).reshape(
                    (self._layers[layer + 1], self._layers[layer])))
            iterator += self._layers[layer] * self._layers[layer + 1]

            network_params.append(np.array([loaded_params[iterator: iterator + self._layers[layer + 1]]]).reshape(
                (self._layers[layer + 1],)))
            iterator += self._layers[layer + 1]

        network_state_dict.values()

        for iterator, key in enumerate(network_state_dict):
            network_state_dict[key] = torch.Tensor(network_params[iterator]).to(device)

        self.load_state_dict(network_state_dict, strict=True)

    def get_device(self):
        network_state_dict = self.state_dict()
        if list(network_state_dict.values())[0].is_cuda:
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")

        return device
