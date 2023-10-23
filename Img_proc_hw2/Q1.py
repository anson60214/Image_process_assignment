import cv2
import numpy as np

# Read the image as GRAYSCALE
Q1_img = cv2.imread('text-broken.tif', cv2.IMREAD_GRAYSCALE)

# SE
ker = np.ones((3, 3), np.uint8)

# Img Binary
_, bin_img = cv2.threshold(Q1_img, 128, 255, cv2.THRESH_BINARY)

# Dilation
dilated_img = cv2.dilate(bin_img, ker, iterations=1)

# Boundary Extraction
boundaries_img = cv2.Canny(dilated_img, 100, 200)

cv2.imwrite('Q1/fixed.tif', dilated_img)
cv2.imwrite('Q1/boundaries.tif', boundaries_img )
