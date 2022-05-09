import torch
import torch.nn as nn


class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def get_trainable_parameters(self):
        parameters = 0

        for params in list(self.parameters()):
            elements = 1

            for size in list(params.size()):
                elements *= size

            parameters += elements

        return parameters

    def get_device(self):
        network_state_dict = self.state_dict()
        if list(network_state_dict.values())[0].is_cuda:
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")

        return device


class EncoderNetwork(BaseNetwork):
    def __init__(self, dropout=0.2):
        super(EncoderNetwork, self).__init__()
        self._dropout = dropout

        self._convolution_block = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=2),
            nn.LeakyReLU(),
            nn.Dropout2d(p=self._dropout),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2),
            nn.LeakyReLU(),
            nn.Dropout2d(p=self._dropout),
            nn.Conv2d(in_channels=32, out_channels=48, kernel_size=3, stride=2),
            nn.LeakyReLU(),
            nn.Dropout2d(p=self._dropout),
            nn.Conv2d(in_channels=48, out_channels=64, kernel_size=3, stride=2),
            nn.LeakyReLU(),
            nn.Dropout2d(p=self._dropout)
        )

        self._dense_block = nn.Sequential(
            nn.Linear(in_features=64 * 4 * 4, out_features=256),
            nn.Tanh(),
            nn.Dropout(p=self._dropout),
            nn.Linear(in_features=256, out_features=96)
        )

    def forward(self, t):
        return self._dense_block(self._convolution_block(t).view(-1, 64 * 4 * 4))


class DecoderNetwork(BaseNetwork):
    def __init__(self, dropout=0.2):
        super(DecoderNetwork, self).__init__()
        self._dropout = dropout

        self._dense_block = nn.Sequential(
            nn.Linear(in_features=96, out_features=256),
            nn.Tanh(),
            nn.Dropout(p=self._dropout),
            nn.Linear(in_features=256, out_features=64 * 4 * 4),
            nn.Tanh(),
            nn.Dropout(p=self._dropout)
        )

        self._deconvolution_block = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=48, kernel_size=3, stride=2),
            nn.LeakyReLU(),
            nn.Dropout2d(p=self._dropout),
            nn.ConvTranspose2d(in_channels=48, out_channels=32, kernel_size=4, stride=2),
            nn.LeakyReLU(),
            nn.Dropout2d(p=self._dropout),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=5, stride=2),
            nn.LeakyReLU(),
            nn.Dropout2d(p=self._dropout),
            nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=7, stride=2),
            nn.Tanh()
        )

    def forward(self, t):
        return self._deconvolution_block(self._dense_block(t).view(-1, 64, 4, 4))


if __name__ == '__main__':
    encoder = EncoderNetwork()
    decoder = DecoderNetwork()

    print(decoder(encoder(torch.rand(100, 1, 91, 91))).shape)
