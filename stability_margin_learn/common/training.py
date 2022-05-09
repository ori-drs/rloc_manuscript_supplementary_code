import torch
import torch.nn as nn

import itertools


def train(training_dataloader, validation_dataloader, device, optimizer, network, writer, batch_size=64, epochs=8,
          evaluate_steps=50, save_path=None):
    iterator = 0

    for epoch in range(epochs):
        training_loss = 0.0
        validation_loss = 0.0

        sub_epoch_iterator = 0

        # Train and Validate
        for training_iterable, validation_iterable in itertools.izip(training_dataloader,
                                                                       itertools.cycle(validation_dataloader)):
            training_batch, training_targets = training_iterable[0].to(device), training_iterable[1].to(device)
            network.train()

            optimizer.zero_grad()
            output = network(training_batch)

            loss = nn.MSELoss()(output, training_targets)

            loss.backward()
            training_loss += (loss.item() / batch_size)

            optimizer.step()

            # Validation Loop
            with torch.no_grad():
                validation_batch, validation_targets = validation_iterable[0].to(device), validation_iterable[1].to(
                    device)
                network.eval()

                output = network(validation_batch)
                loss = nn.MSELoss()(output, validation_targets)

                validation_loss += (loss.item() / batch_size)

                if sub_epoch_iterator % evaluate_steps == evaluate_steps - 1:  # print every 50 mini-batches
                    print(
                            '[Training, %d, %5d] loss: %.8f' % (
                        epoch + 1, sub_epoch_iterator + 1, training_loss / evaluate_steps)
                    )
                    print(
                            '[Validation, %d, %5d] loss: %.8f' % (
                        epoch + 1, sub_epoch_iterator + 1, validation_loss / evaluate_steps)
                    )

                    writer.add_scalars('Loss', {'Training': training_loss / evaluate_steps}, iterator)
                    writer.add_scalars('Loss', {'Validation': validation_loss / evaluate_steps}, iterator)

                    iterator += 1
                    training_loss = 0.0
                    validation_loss = 0.0

                sub_epoch_iterator += 1

    if save_path is not None:
        torch.save(network.state_dict(), save_path)

    return iterator
