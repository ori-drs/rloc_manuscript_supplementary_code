import numpy as np

import torch

from common.paths import ProjectPaths
from common.networks import EncoderNetwork


def main():
    paths = ProjectPaths()

    model_directory = '/path/to/trained/model/'
    model_directory = paths.TRAINED_MODELS_PATH + model_directory

    network = EncoderNetwork()
    network.load_state_dict(torch.load(model_directory + 'encoder_state_dict.pt'))

    network.cpu().eval()

    network_input = np.zeros((91, 91))
    network_input[0, :10] = np.ones(10)

    print('Network Input:\n', network_input)
    print('\n')
    print('Network Output:\n', network(torch.from_numpy(network_input).view((1, 1, 91, 91)).float()))


if __name__ == '__main__':
    main()
