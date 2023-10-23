import cv2
import numpy as np

# salt_per and per_per is probability in range of [0,1]
def salt_pepper_noise(image, salt_per, pepper_per):

    result = np.copy(image)

    # Salt
    num_salt = np.ceil(salt_per * image.size)
    coords = [np.random.randint(0, i - 1, int(num_salt))
          for i in image.shape]
    result[tuple(coords)] = 255

    # Pepper
    num_pepper = np.ceil(pepper_per * image.size)
    coords = [np.random.randint(0, i - 1, int(num_pepper))
          for i in image.shape]
    result[tuple(coords)] = 0
    return result

# Different noise levels
noise_levels = [0.1, 0.3, 0.5, 0.7, 0.9]


image = cv2.imread('baboon.bmp', 0)  # Read the image in grayscale mode
for noise_level in noise_levels:
    # Add the noise to the image
    noisy_image = salt_pepper_noise(image, noise_level, noise_level)

    # Save image
    cv2.imwrite("Q1a/"+f'baboon_{int(noise_level*100)}per_noise.bmp', noisy_image)

image = cv2.imread('peppers.bmp', 0)  # Read the image in grayscale mode
for noise_level in noise_levels:
    # Add the noise to the image
    noisy_image = salt_pepper_noise(image, noise_level, noise_level)

    # Save image
    cv2.imwrite("Q1a/"+f'peppers_{int(noise_level*100)}per_noise.bmp', noisy_image)
