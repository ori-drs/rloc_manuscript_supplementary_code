import numpy as np

import torch
import torch.nn.functional as F

from common.paths import ProjectPaths
from common.networks import MultiLayerPerceptron


def main():
    paths = ProjectPaths()

    model_directory = '/enter/trained/model/directory/'
    model_directory = paths.TRAINED_MODELS_PATH + model_directory

    network = MultiLayerPerceptron(in_dim=47, out_dim=1, hidden_layers=[256, 128, 128], activation=F.softsign, dropout=0.0)
    network.load_state_dict(torch.load(model_directory + 'network_state_dict.pt'))

    model_parameters = list(network.state_dict().keys())
    model_parameters = np.concatenate(
        [network.state_dict()[key].cpu().numpy().transpose().reshape(-1) for key in model_parameters])

    param_save_name = model_directory + 'network_parameters_anymal_c.txt'
    np.savetxt(param_save_name, model_parameters.reshape((1, -1)), delimiter=', ',
               newline='\n', fmt='%1.10f')

    print '\nSaved model parameters in the following order:'
    for parameter_key in list(network.state_dict().keys()):
        print '   ', parameter_key, '| Dimension:', network.state_dict()[parameter_key].shape


if __name__ == '__main__':
    main()
