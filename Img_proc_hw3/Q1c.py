import cv2
import numpy as np
import csv
from skimage.util import random_noise
from skimage.metrics import peak_signal_noise_ratio as psnr
from scipy.ndimage import gaussian_filter

def salt_pepper_noise(image, noise_ratio):
    noisy = random_noise(image, mode='s&p', salt_vs_pepper=0.5, amount=noise_ratio)
    noisy = np.array(255*noisy, dtype = 'uint8')
    return noisy

def gaussian_filter_image(image, sigma):
    filtered_image = gaussian_filter(image, sigma=sigma)
    return filtered_image

if __name__ == '__main__':
    images = ['baboon.bmp', 'peppers.bmp']
    noise_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    with open("Q1/" + 'result_c.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Image', 'Noise Ratio', 'Before PSNR', 'Gaussian Filter PSNR'])
        for image_path in images:
            original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            for noise_ratio in noise_ratios:
                noisy = salt_pepper_noise(original, noise_ratio)
                gaussian_filtered = gaussian_filter_image(noisy, 2)

                psnr_before = np.round(psnr(original, noisy),1)
                psnr_gaussian = np.round( psnr(original, gaussian_filtered),1)

                writer.writerow([image_path, noise_ratio, psnr_before, psnr_gaussian])
