import math
import random
import warnings

import imageio
import numpy as np

from scipy.ndimage import rotate
from skimage.transform import resize


class ImageDatasetHandler:
    def __init__(self):
        self._images = []

    def __len__(self):
        return len(self._images)

    def __getitem__(self, index):
        return self._images[index]

    def append(self, file_name):
        if file_name.endswith('.png'):
            self._images.append(imageio.imread(file_name))
        else:
            warnings.warn('Unsupported file type')

    def shuffle(self):
        random.shuffle(self._images)

    def generate_augmented_samples(self, samples, ratio=1, crop=None):
        image_indices = np.random.randint(self.__len__(), size=max(1, math.ceil(samples / ratio)))
        image_samples = []

        for index in image_indices:
            image = self.__getitem__(index)

            for augmented_sample in range(ratio):
                image_shape = list(image.shape)
                processed_image = np.array(image, dtype=np.float)

                if image_shape[0] >= 2.0 * crop[0] and image_shape[1] >= 2.0 * crop[1] and ratio > 1:
                    if random.random() < 0.5:
                        scale = (random.random() * 3) + 1

                        image_shape[0] = int(image_shape[0] / scale)
                        image_shape[1] = int(image_shape[1] / scale)

                        # Ensure the image obtained after scaling is greater than cropped image
                        if crop is not None:
                            while image_shape[0] < crop[0] and image_shape[1] < crop[1]:
                                scale = random.random() + 1
                                image_shape[0] = int(image_shape[0] / scale)
                                image_shape[1] = int(image_shape[1] / scale)

                        processed_image = resize(processed_image, image_shape)

                    if random.random() < 0.1:
                        processed_image = rotate(processed_image, angle=(random.random() - 0.5) * 180, reshape=False)

                if crop is not None:
                    x = random.randint(0, image_shape[0] - crop[0])
                    y = random.randint(0, image_shape[1] - crop[1])

                    processed_image = processed_image[x: x + crop[0], y: y + crop[1]]

                if random.random() < 0.5 and ratio > 1:
                    processed_image = np.flip(processed_image, 0)

                if random.random() < 0.5 and ratio > 1:
                    processed_image = np.flip(processed_image, 1)

                if random.random() < 0.5 and ratio > 1:
                    processed_image = np.transpose(processed_image)

                if random.random() < 0.5 and ratio > 1:
                    processed_image = np.clip(processed_image * random.random() * 2, 0, 65535)

                image_samples.append(processed_image)

                if len(image_samples) >= samples:
                    break

        random.shuffle(image_samples)
        return np.array(image_samples)
