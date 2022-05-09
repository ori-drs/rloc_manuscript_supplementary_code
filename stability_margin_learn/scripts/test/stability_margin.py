import numpy as np

import torch
import torch.nn.functional as F

from common.paths import ProjectPaths
from common.datasets import TrainingDataset
from common.networks import MultiLayerPerceptron


def main():
    paths = ProjectPaths()
    dataset_handler = TrainingDataset()

    model_directory = '/enter/trained/model/directory/'
    model_directory = paths.TRAINED_MODELS_PATH + model_directory

    network = MultiLayerPerceptron(in_dim=47, out_dim=1, hidden_layers=None, activation=F.softsign, dropout=0.0)
    network.load_state_dict(torch.load(model_directory + 'network_state_dict.pt'))

    network.eval()

    network_input = np.concatenate([
        np.array([0.0, 0.0, 1.0]),
        np.zeros(15),
        np.array([0.36, 0.21, -0.47]),
        np.array([0.36, -0.21, -0.47]),
        np.array([-0.36, 0.21, -0.47]),
        np.array([-0.36, -0.21, -0.47]),
        np.array([0.5]),
        np.ones(4),
        np.array([0.0, 0.0, 1.0] * 4)
    ])

    network_input = dataset_handler.scale_data(network_input, input_only=True)
    print 'Stability Margin: ', 0.1 * network(torch.from_numpy(network_input).float()).item()


if __name__ == '__main__':
    main()
