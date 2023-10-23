import cv2
import csv
import numpy as np
import pandas as pd
from skimage.util import random_noise
from skimage.metrics import peak_signal_noise_ratio as psnr


def salt_pepper_noise(image, noise_ratio):
    noisy = random_noise(image, mode='s&p', salt_vs_pepper=0.5, amount=noise_ratio)
    noisy = np.array(255*noisy, dtype = 'uint8')
    return noisy

def mean_filter(image, kernel_size):
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
    filtered_image = cv2.filter2D(image, -1, kernel)
    filtered_image[(image == 0) | (image == 255)] = image[(image == 0) | (image == 255)]
    return filtered_image

if __name__ == '__main__':
    images = ['baboon.bmp', 'peppers.bmp']
    noise_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]

    with open('Q1/result_b.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Image', 'Noise Ratio', 'Before PSNR', 'Mean Filter PSNR'])
        for image_path in images:
            original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            for noise_ratio in noise_ratios:
                noisy =salt_pepper_noise(original, noise_ratio)
                mean_filtered = mean_filter(noisy, 5)

                psnr_before = np.round( psnr(original, noisy),1)
                psnr_mean = np.round( psnr(original, mean_filtered),1)

                writer.writerow([image_path, noise_ratio, psnr_before, psnr_mean])
