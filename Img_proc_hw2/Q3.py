import cv2
import numpy as np

img1 = 'aerialview-washedout.tif'
img2 = 'einstein-low-contrast.tif'

# Read the image as GRAYSCALE
img1 = cv2.imread(img1, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(img2, cv2.IMREAD_GRAYSCALE)

def histogram(img):
   hist = [0.0]*256
   height, width = img.shape
   for i in range(height):
        for j in range(width):
            intensity = img[i, j]
            hist[intensity] += 1
   return hist

def normalization(cdf):
    return (cdf - np.min(cdf))*255 / (np.max(cdf) - np.min(cdf))

def global_HE(img):

    height, width = img.shape
    # Calculate the histogram
    hist = histogram(img)

    # Calculate the cumulative distribution function (CDF)
    cdf = np.zeros(256, dtype=int)
    cdf[0] = hist[0]
    for i in range(1, 256):
        cdf[i] = cdf[i-1] + hist[i]

    # Normalize the CDF
    cdf_norm = normalization(cdf)

    # Map intensity values using the normalized CDF
    eq_img = np.zeros_like(img)
    for i in range(height):
        for j in range(width):
            intensity = img[i, j]
            eq_img[i, j] = cdf_norm[intensity]
    
    return eq_img

cv2.imwrite('Q3/aerialview-washedout-fixed.tif', global_HE(img1))
cv2.imwrite('Q3/einstein-low-contrast-fixed.tif', global_HE(img2))
