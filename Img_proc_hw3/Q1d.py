import cv2
import numpy as np
from skimage.util import random_noise
from skimage.metrics import peak_signal_noise_ratio as psnr
import csv

def add_salt_pepper_noise(image, noise_ratio):
    noisy = random_noise(image, mode='s&p', salt_vs_pepper=0.5, amount=noise_ratio)
    noisy = np.array(255*noisy, dtype = 'uint8')
    return noisy

def adaptive_median_filter(noisy, max_kernel_size=11):
    height, width = noisy.shape
    padded_image = np.pad(noisy, max_kernel_size//2, mode='constant')
    filtered = noisy.copy()

    for i in range(height):
        for j in range(width):
            kernel_size = 3
            while kernel_size <= max_kernel_size:
                kernel_start = max_kernel_size//2 - kernel_size//2
                kernel_end = max_kernel_size//2 + kernel_size//2
                sub_image = padded_image[i+kernel_start:i+kernel_end+1, j+kernel_start:j+kernel_end+1]
                sorted_pixels = np.sort(sub_image.flatten())
                median = sorted_pixels[(kernel_size**2)//2]

                if median != 0 and median != 255:
                    filtered[i, j] = median
                    break

                kernel_size += 2

    return filtered

def main():
    images = ['baboon.bmp', 'peppers.bmp']
    noise_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    with open("Q1/" + 'result_d.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Image', 'Noise Ratio', 'Before PSNR', 'Adaptive Median Filter PSNR'])
        for image_path in images:
            original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            for noise_ratio in noise_ratios:
                noisy = add_salt_pepper_noise(original, noise_ratio)
                filtered = adaptive_median_filter(noisy)

                psnr_before = np.round(psnr(original, noisy),1)
                psnr_after = np.round( psnr(original, filtered),1)

                writer.writerow([image_path, noise_ratio, psnr_before, psnr_after])

if __name__ == '__main__':
    main()