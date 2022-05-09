import os

import torch
import torch.optim as optim

from tensorboard import program
from torch.utils.tensorboard import SummaryWriter

import socket
import webbrowser

from common.training import Trainer
from common.paths import ProjectPaths
from common.datasets import ImageDatasetHandler
from common.networks import EncoderNetwork, DecoderNetwork

BATCH_SIZE = 128
DATA_AUGMENTATION_RATIO = 16
LEARNING_RATE = 0.001
SAVE_TRACED_MODEL = True
EVALUATE_STEPS = 100
MAX_IMAGE_FILES = 25000
MAX_TRAINING_STEPS = 5000
LAUNCH_TENSORBOARD = True
LOAD_STATE_DICT = None
STATE_DICT_PATH = '/path/to/trained/model/checkpoint/'


def main():
    # Get training data
    data_folder = 'height_maps_bricks'
    image_dataset = ImageDatasetHandler()

    paths = ProjectPaths()

    num_files = 0
    for path, _, files in os.walk(paths.DATA_PATH + '/' + data_folder):
        for file_name in files:
            if len(image_dataset) < MAX_IMAGE_FILES:
                image_dataset.append(os.path.join(path, file_name))
            else:
                break
            num_files += 1
            print('\rFiles Processed: {}'.format(num_files), end='')
        if len(image_dataset) >= MAX_IMAGE_FILES:
            break
    print()

    # Shuffle the images in the image_dataset list
    image_dataset.shuffle()

    # Use CUDA if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Encoder and decoder networks
    encoder = EncoderNetwork(dropout=0.0).to(device)
    decoder = DecoderNetwork(dropout=0.0).to(device)

    if LOAD_STATE_DICT:
        encoder.load_state_dict(torch.load(paths.TRAINED_MODELS_PATH + STATE_DICT_PATH + 'encoder_state_dict.pt'))
        decoder.load_state_dict(torch.load(paths.TRAINED_MODELS_PATH + STATE_DICT_PATH + 'decoder_state_dict.pt'))

    # Optimization technique
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=LEARNING_RATE)

    # TensorBoard
    log_path = paths.LOGS_PATH + '/' + data_folder + '/denoising_autoencoder/' + paths.INIT_DATETIME_STR
    writer = SummaryWriter(log_path)

    if LAUNCH_TENSORBOARD:
        tensorboard_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        tensorboard_socket.bind(('', 0))
        tensorboard_port = tensorboard_socket.getsockname()[1]
        tensorboard_socket.close()

        tensorboard_launcher = program.TensorBoard()
        tensorboard_launcher.configure(
            argv=[None, '--logdir', log_path, '--port', str(tensorboard_port)])
        tensorboard_address = tensorboard_launcher.launch()
        webbrowser.open_new_tab(tensorboard_address + '#scalars&_smoothingWeight=0')

    # Network Save Path
    model_save_dir = paths.TRAINED_MODELS_PATH + '/' + data_folder + '/denoising_autoencoder/' + paths.INIT_DATETIME_STR
    os.makedirs(model_save_dir, exist_ok=True)

    # Call the train method
    trainer = Trainer()
    trainer.train(image_dataset=image_dataset, encoder=encoder, decoder=decoder, device=device, optimizer=optimizer,
                  writer=writer, batch_size=BATCH_SIZE, augmentation_ratio=DATA_AUGMENTATION_RATIO, image_dim=[91, 91],
                  max_training_steps=MAX_TRAINING_STEPS, evaluate_steps=EVALUATE_STEPS, save_dir=model_save_dir)

    while True:
        try:
            pass
        except KeyboardInterrupt:
            exit(0)


if __name__ == '__main__':
    main()
