import numpy as np
import cv2


# Read the image as 
Q2_img = cv2.imread('einstein-low-contrast.tif', cv2.IMREAD_GRAYSCALE)


def linear_stretching(img):
    # Calculate the minimum and maximum pixel values
    min_val = np.min(img)
    max_val = np.max(img)
    
    # Perform linear stretching
    stretched_img = (img - min_val) * (255.0 / (max_val - min_val))
    # Convert the stretched_img to uint8 data type
    stretched_img = stretched_img.astype(np.uint8)
    
    return stretched_img

cv2.imwrite('Q2/Stretched_img.tif', linear_stretching(Q2_img) )

