from common.paths import ProjectPaths
from common.training import train
from common.networks import MultiLayerPerceptron
from common.datasets import TrainingDataset

import os
import socket
import webbrowser

import torch
import torch.nn as nn
import torch.optim as optim

from tensorboard import program
from torch.utils.tensorboard import SummaryWriter

EPOCHS = 128
BATCH_SIZE = 1024
LEARNING_RATE = 0.0001
SAVE_TRACED_MODEL = False
EVALUATE_STEPS = 50
LAUNCH_TENSORBOARD = True


def main():
    paths = ProjectPaths()

    training_dataset_handler = TrainingDataset()
    data_parser = training_dataset_handler.get_training_data_parser()
    data_folder = training_dataset_handler.get_data_folder()

    # Use torch data_loader to sample training data
    dataloader_params = {'batch_size': BATCH_SIZE, 'shuffle': True, 'num_workers': 12}
    training_dataloader = data_parser.torch_dataloader(dataloader_params, data_type='training')
    validation_dataloader = data_parser.torch_dataloader(dataloader_params, data_type='validation')

    # Initialize Network Object
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    network = MultiLayerPerceptron(hidden_layers=[256, 128, 128])
    network.to(device)

    optimizer = optim.Adam(network.parameters(), lr=LEARNING_RATE)
    print '\nTraining for', network.get_trainable_parameters(), 'network parameters.\n'

    # TensorBoard
    log_path = paths.LOGS_PATH + '/' + data_folder + '/' + paths.INIT_DATETIME_STR
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
    model_save_dir = paths.TRAINED_MODELS_PATH + '/' + data_folder + '/' + paths.INIT_DATETIME_STR
    try:
        os.makedirs(model_save_dir)
    except OSError:
        pass
    save_path = model_save_dir + '/network_state_dict.pt'

    # Train and Validate
    iterator_offset = train(training_dataloader, validation_dataloader, device, optimizer, network, writer,
                            BATCH_SIZE, EPOCHS, EVALUATE_STEPS, save_path=save_path)

    if SAVE_TRACED_MODEL:
        save_path = model_save_dir + '/traced_network_model.pt'

        params = {'batch_size': 1, 'shuffle': True, 'num_workers': 1}
        tracing_dataloader = data_parser.torch_dataloader(params=params, data_type='testing')

        for local_batch, local_targets in tracing_dataloader:
            traced_network = torch.jit.trace(network.cpu(), local_batch.cpu())
            break

        traced_network.cpu().save(save_path)

    # Testing
    network.to(device).eval()
    dataloader_params = {'batch_size': max(16, min(int(BATCH_SIZE / EPOCHS), 48)), 'shuffle': True, 'num_workers': 12}
    testing_dataloader = data_parser.torch_dataloader(params=dataloader_params, data_type='testing')

    test_loss = 0.0
    sub_epoch_iterator = 0
    prediction_iterator = 0
    iterator = iterator_offset - 1

    for local_batch, local_targets in testing_dataloader:
        local_batch, local_targets = local_batch.to(device), local_targets.to(device)

        output = network(local_batch)

        loss = nn.MSELoss()(output, local_targets)
        test_loss += (loss.item() / BATCH_SIZE)

        if sub_epoch_iterator % EVALUATE_STEPS == EVALUATE_STEPS - 1:
            print '[Test, %d, %5d] loss: %.8f' % (1, sub_epoch_iterator + 1, test_loss / EVALUATE_STEPS)
            iterator += 1

            writer.add_scalars('Loss', {'Test': test_loss / EVALUATE_STEPS}, iterator)
            test_loss = 0.0

            if prediction_iterator < EVALUATE_STEPS:
                writer.add_scalars('Prediction', {'Target': local_targets[-1].item() / 10.0},
                                   prediction_iterator)
                writer.add_scalars('Prediction', {'NetworkOutput': output[-1].item() / 10.0},
                                   prediction_iterator)
                prediction_iterator += 1

        sub_epoch_iterator += 1

    while True:
        try:
            pass
        except KeyboardInterrupt:
            exit(0)


if __name__ == '__main__':
    main()
