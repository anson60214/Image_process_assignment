import cv2
import numpy as np


# Read the image as GRAYSCALE
img = cv2.imread('aerialview-washedout.tif', cv2.IMREAD_GRAYSCALE)

def histogram(img):
   hist = [0.0]*256
   height, width = img.shape
   for i in range(height):
        for j in range(width):
            intensity = img[i, j]
            hist[intensity] += 1
   return hist

def median(img,hist):
   height, width = img.shape
   total_pixels = height * width 
   median = 0
   cdf = 0
   for i in range(256):
      cdf += hist[i]
      if cdf >= int(total_pixels / 2):
         median = i
         break
   return median

def median_HE(img): 
   height, width = img.shape
   
   # Calculate the histogram
   hist = histogram(img)

   # Find the median value
   med = median(img,hist)

   # Perform histogram equalization on the low sub-histogram (0 ~ median)
   cdf_low = 0
   for i in range(med + 1):
      cdf_low += hist[i]

   cdf_low_norm = []
   cdf = 0
   for i in range(med + 1):
      cdf += hist[i]
      cdf_low_norm.append(int(cdf*255 / cdf_low))

   # Perform histogram equalization on the high sub-histogram ((median + 1) ~ 255)
   cdf_high = 0
   for i in range(med + 1, 256):
      cdf_high += hist[i]

   cdf_high_norm = []
   cdf = 0
   for i in range(med + 1, 256):
      cdf += hist[i]
      cdf_high_norm.append(int(((cdf-1)*(255-med)) / cdf_high) + med + 1)

   # Map intensity values using the equalized sub-histograms
   median_HE_img = img.copy()
   for i in range(height):
      for j in range(width):
         intensity = img[i, j]
         if intensity <= med:
               median_HE_img[i, j] = cdf_low_norm[intensity]
         else:
               median_HE_img[i, j] = cdf_high_norm[intensity - (med + 1)]
            
   return median_HE_img


cv2.imwrite('Q4/aerialview-washedout-fixed.tif', median_HE(img) )