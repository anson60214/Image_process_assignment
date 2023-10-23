import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import copy
import math
from scipy.fftpack import dct, idct
import cv2

# Q1
# read png
left = Image.open('laptop_left.png')
right = Image.open('laptop_right.png')
left_data = np.array(left)
right_data = np.array(right)

def combine_img(img1,img2):
    # get xyz
    left_x,left_y,left_z= img1.shape
    right_x,right_y,right_z= img2.shape

    # Create a new matrix to hold the concatenated data
    result1 = np.zeros((left_x, left_y+right_y, left_z))

    # Copy the data from the first matrix into the new matrix
    result1[:, :left_y, :] = img1

    # Copy the data from the second matrix into the new matrix
    result1[:, left_y:(left_y+right_y), :] = img2

    # Convert the matrix to an image
    img1 = Image.fromarray(result1.astype('uint8'))
    return img1

img1 = combine_img(left_data,right_data)
# Save the image to a file
img1.save('Q1_result.png')

#--------------------------------------------------------------------------#

# Q2

def rotate_img(image, angle):
    # Convert angle to radians
    theta = math.radians(angle)
    
    # Calculate sine and cosine of angle
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    rotation_matrix = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])


    # Get dimensions of image
    width, height = image.size
    
    # Create new image object with same dimensions as original image
    rotated_image = Image.new('RGB', (width, height))
    
    # Loop through each pixel in the rotated image
    for x in range(width):
        for y in range(height):
            xy = [x,y]
            # Calculate new pixel position
            new_xy = np.matmul(xy, rotation_matrix)

            new_x = new_xy[0]
            new_y = new_xy[1]
            
            # Check if new pixel position is within bounds of original image
            if new_x >= 0 and new_x < width and new_y >= 0 and new_y < height:
                # Get pixel value from original image and set it in the rotated image
                pixel = image.getpixel((new_x, new_y))
                rotated_image.putpixel((x, y), pixel)
    
    return rotated_image

img2 = rotate_img(img1,15)
# Save rotated image
img2.save('Q2_result.png')

#--------------------------------------------------------------------------#

# Q3

def resize_image(image, new_width, new_height):
    # Get the current dimensions of the image
    width, height, channels = image.shape
    
    # Calculate the scaling factor for each dimension
    x_scale = float(new_width) / width
    y_scale = float(new_height) / height
    
    # Create a new array to hold the resized image
    resized_image = np.zeros((new_height, new_width, channels), dtype=np.uint8)
    
    # Iterate over each pixel in the resized image
    for y in range(new_height):
        for x in range(new_width):
            # Calculate the corresponding pixel in the original image
            x_orig = x / x_scale
            y_orig = y / y_scale
            
            # Calculate the four nearest pixels in the original image
            x1 = int(x_orig)
            x2 = min(x1 + 1, width - 1)
            y1 = int(y_orig)
            y2 = min(y1 + 1, height - 1)
            
            # Calculate the distances to each of the four nearest pixels
            x_dist = x_orig - x1
            y_dist = y_orig - y1
            
            # Calculate the area of 4 regions
            A1 = (1 - x_dist) * (1 - y_dist)
            A2 = x_dist * (1 - y_dist)
            A3 = (1 - x_dist) * y_dist
            A4 = x_dist * y_dist

            # Get the values of the four nearest pixels
            f11 = image[y1, x1]
            f21 = image[y1, x2]
            f12 = image[y2, x1]
            f22 = image[y2, x2]
            
            # Interpolate the pixel value using bilinear interpolation
            pixel_value = A1 * f11 + A2 * f21 + A3 * f12 + A4 * f22
            
            # Set the pixel value in the resized image
            resized_image[y, x] = pixel_value
    
    return resized_image

lena = Image.open('lena.bmp')
lena_data = np.array(lena)
result3 = resize_image(lena_data,1024,1024)
# Convert the matrix to an image
img3 = Image.fromarray(result3.astype('uint8'))

# Save the image to a file
img3.save('Q3_result.png')

#--------------------------------------------------------------------------#

# Q4

graveler = Image.open('graveler.bmp')
graveler_data = np.array(graveler)
width, height, channels = graveler_data.shape

overlay_lena = copy.deepcopy(result3)
graveler_list = list(graveler_data)

for i in range(width):
    for j in range(height):
        if tuple(graveler_list[i][j])!=(255, 255, 255):
            overlay_lena[i][j] = graveler_list[i][j]
overlay = Image.fromarray(np.uint8(overlay_lena))

overlay.save('Q4_result.png')

#--------------------------------------------------------------------------#

# Q5

from scipy.fftpack import dct, idct

def dct2(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def idct2(block):
    return idct(idct(block.T, norm='ortho').T, norm='ortho')

def embedwatermark(image, watermark, alpha=0.1):
    h, w, _  = image.shape
    watermark_resized = cv2.resize(watermark, (w, h), interpolation=cv2.INTER_LINEAR)

    coeffs_image = dct2(image[:, :, 0])
    coeffs_watermark = dct2(watermark_resized)

    watermarked_coeffs = coeffs_image + alpha * coeffs_watermark
    watermarked_y_channel = idct2(watermarked_coeffs)
    watermarked_img = image.copy()
    watermarked_img[:, :, 0] = watermarked_y_channel
    return watermarked_img

def extract_watermark(image, original_image, alpha=1):
    coeffs_watermarked_image = dct2(image[:, :, 0])
    coeffs_original_image = dct2(original_image[:, :, 0])

    extracted_coeffs = (coeffs_watermarked_image - coeffs_original_image) / alpha
    extracted_watermark = idct2(extracted_coeffs)

    return extracted_watermark

lena_img = cv2.imread("lena.bmp")
flipped_lena = cv2.flip(lena_img, 1)
# Save the flipped lena
cv2.imwrite("Q5a0_result.png", flipped_lena)

flipped_lena_ycrcb = cv2.cvtColor(flipped_lena, cv2.COLOR_BGR2YCrCb)

# Load watermark image and convert it to grayscale
watermark = cv2.imread("graveler.bmp")
watermark_gray = cv2.cvtColor(watermark, cv2.COLOR_BGR2GRAY)

# Embed the watermark into the Lena image
watermarked_img_ycrcb = embedwatermark(flipped_lena_ycrcb, watermark_gray)
watermarked_img = cv2.cvtColor(watermarked_img_ycrcb, cv2.COLOR_YCrCb2BGR)

# Save the watermarked image
cv2.imwrite("Q5a1_result.png", watermarked_img)

# Extract the watermark from the watermarked image
extracted_watermark = extract_watermark(watermarked_img_ycrcb, flipped_lena_ycrcb)
cv2.imwrite("Q5a2_result.png", extracted_watermark)
