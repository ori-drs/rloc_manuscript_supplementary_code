import numpy as np

import torch

from common.paths import ProjectPaths
from common.networks import MultiLayerPerceptron


def main():
    paths = ProjectPaths()

    model_directory = '/path/to/trained/model/'
    model_directory = paths.TRAINED_MODELS_PATH + model_directory

    network = MultiLayerPerceptron()
    network.load_state_dict(torch.load(model_directory + 'network_state_dict.pt'))

    network.eval()

    network_input = np.zeros(24)

    print('Network Output: ', network(torch.from_numpy(network_input).float()).item() / 0.023)


if __name__ == '__main__':
    main()
