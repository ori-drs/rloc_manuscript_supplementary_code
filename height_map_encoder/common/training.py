import os
import random
from threading import Thread

import torch
import torch.nn as nn
import torch.nn.functional as functional


def impulse_noise(image, prob):
    '''
    Based on: https://github.com/kimhc6028/pathnet-pytorch/blob/master/common.py
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    noise = torch.rand(image.shape)
    image[noise < prob * 0.5] = -1.
    image[noise > 1 - (prob * 0.5)] = 1.

    return image


class Trainer:
    def __init__(self):
        self._training_samples = None
        self._blurring_kernel = torch.Tensor([[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]]) * (1. / 9.)

    def train(self, image_dataset, encoder, decoder, device, optimizer, writer=None, batch_size=64,
              augmentation_ratio=10, image_dim=None, max_training_steps=1000, evaluate_steps=50, save_dir=None,
              save_interval=None):
        if image_dim is None:
            image_dim = [91, 91]

        self._blurring_kernel = self._blurring_kernel.view(1, 1, image_dim[0], image_dim[1]).repeat(batch_size, 1, 1, 1)

        training_steps = 0
        training_loss = 0
        evaluate_iterator = 0

        # Start an instance of the sample generator
        sample_generator_thread = Thread(target=self.generate_training_sample,
                                         args=(image_dataset, batch_size, augmentation_ratio, image_dim,))
        sample_generator_thread.start()

        while training_steps < max_training_steps:
            sample_generator_thread.join()
            training_samples = (torch.tensor(self._training_samples, requires_grad=False, dtype=torch.float).view(
                batch_size, 1, image_dim[0], image_dim[1]).to(device) - 32767.5) * (1.0 / 32767.5)

            # Start another instance of the sample generator
            sample_generator_thread = Thread(target=self.generate_training_sample,
                                             args=(image_dataset, batch_size, augmentation_ratio, image_dim,))
            sample_generator_thread.start()

            # Ensure the networks are in training mode
            encoder.train()
            decoder.train()

            optimizer.zero_grad()

            encoder_input = training_samples.clone().detach()
            encoder_input_n = 0.05 * abs(random.random()) * (
                        training_steps / max_training_steps) * torch.randn(batch_size, 1, image_dim[0],
                                                                           image_dim[1]).float().to(device)
            encoder_input_i = impulse_noise(encoder_input, 0.05 * abs(random.random())) - encoder_input
            encoder_input_c = abs(random.random()) * (functional.conv2d(encoder_input, self._blurring_kernel, padding='same') - encoder_input)

            encoder_input = encoder_input_n + encoder_input_i + encoder_input_c

            reconstruction = decoder(encoder(encoder_input))
            loss = nn.MSELoss()(reconstruction, training_samples)

            loss.backward()
            optimizer.step()

            training_steps += 1
            training_loss += (loss.item() / batch_size)

            if training_steps % evaluate_steps == evaluate_steps - 1:
                print('[Training, %d] loss: %.8f' % (training_steps + 1, training_loss / evaluate_steps))

                writer.add_image('Decoder Reconstruction', (reconstruction[0] + 1.0) * 0.5, 0)
                writer.add_image('Encoder Input', (encoder_input[0] + 1.0) * 0.5, 0)
                writer.add_image('Original Image', (training_samples[0] + 1.0) * 0.5, 0)

                if writer is not None:
                    writer.add_scalar('Loss/Training', training_loss / evaluate_steps, evaluate_iterator)

                training_loss = 0
                evaluate_iterator += 1

            if save_interval is not None:
                if training_steps % save_interval == save_interval - 1 and save_dir is not None:
                    model_save_dir = save_dir
                    save_dir = model_save_dir + '/' + str(int(training_steps / save_interval))
                    os.makedirs(save_dir, exist_ok=True)

                    torch.save(encoder.state_dict(), save_dir + '/encoder_state_dict.pt')
                    torch.save(decoder.state_dict(), save_dir + '/decoder_state_dict.pt')

                    save_path = save_dir + '/traced_encoder_model.pt'
                    traced_network = torch.jit.trace(encoder.eval().cpu(), training_samples.cpu())
                    traced_network.cpu().save(save_path)

                    save_path = save_dir + '/traced_decoder_model.pt'
                    traced_network = torch.jit.trace(decoder.eval().cpu(), encoder(training_samples.cpu()))
                    traced_network.cpu().save(save_path)

                    encoder.train().to(device)
                    decoder.train().to(device)
                    save_dir = model_save_dir

        if save_dir is not None:
            torch.save(encoder.state_dict(), save_dir + '/encoder_state_dict.pt')
            torch.save(decoder.state_dict(), save_dir + '/decoder_state_dict.pt')

            save_path = save_dir + '/traced_encoder_model.pt'
            traced_network = torch.jit.trace(encoder.eval().cpu(), training_samples.cpu())
            traced_network.cpu().save(save_path)

            save_path = save_dir + '/traced_decoder_model.pt'
            traced_network = torch.jit.trace(decoder.eval().cpu(), encoder(training_samples.cpu()))
            traced_network.cpu().save(save_path)

    def generate_training_sample(self, image_dataset, samples, ratio, crop):
        self._training_samples = image_dataset.generate_augmented_samples(samples=samples, ratio=ratio, crop=crop)
