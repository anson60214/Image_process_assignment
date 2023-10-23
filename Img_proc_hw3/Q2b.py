import cv2
import numpy as np
import matplotlib.pyplot as plt

def laplacian_filter(img):
    # Define the Laplacian kernel
    kernel = np.array([[0, 1, 0],
                       [1, -4, 1],
                       [0, 1, 0]])

    # Get image shape
    rows, cols = img.shape

    # Initialize empty output array
    filtered = np.zeros_like(img, dtype=np.float32)

    # Apply Laplacian filter
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            patch = img[i - 1:i + 2, j - 1:j + 2]
            filtered[i, j] = np.sum(patch * kernel)

    # Threshold the filtered result to obtain the edge map
    threshold_value = 0  # Adjust this value to change the threshold
    edges = np.where(filtered > threshold_value, 255, 0).astype(np.uint8)

    return edges

def sobel_filter(img):
    # Define the Sobel kernels
    kernel_x = np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]])

    kernel_y = np.array([[-1, -2, -1],
                         [0, 0, 0],
                         [1, 2, 1]])

    # Get image shape
    rows, cols = img.shape

    # Initialize empty output arrays
    gradient_x = np.zeros_like(img, dtype=np.float32)
    gradient_y = np.zeros_like(img, dtype=np.float32)

    # Apply Sobel filter
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            gradient_x[i, j] = np.sum(img[i-1:i+2, j-1:j+2] * kernel_x)
            gradient_y[i, j] = np.sum(img[i-1:i+2, j-1:j+2] * kernel_y)

    # Calculate the gradient magnitude and direction
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    gradient_direction = np.arctan2(gradient_y, gradient_x) * 180 / np.pi

    return gradient_magnitude, gradient_direction


# Read the image1
img = cv2.imread("peppers.bmp", cv2.IMREAD_GRAYSCALE)


# Apply Gaussian blur to smooth the image
img_blurred = cv2.GaussianBlur(img, (3, 3), 1)

# Apply Sobel filter
gradient_magnitude, gradient_direction = sobel_filter(img_blurred)

# Threshold the gradient magnitude to obtain the edge map
threshold_value = 50  # Adjust this value to change the threshold
edges = np.where(gradient_magnitude > threshold_value, 255, 0).astype(np.uint8)

# Display the resulting edge map
plt.imshow(edges, cmap='gray')
plt.axis('off')
plt.savefig("Q2b/Q2b_1")

# Apply Laplacian filter for edge detection
edges1 = laplacian_filter(img)

# Display the resulting edge map
plt.imshow(edges1, cmap='gray')
plt.axis('off')
plt.savefig("Q2b/Q2b_3")

# Read the image2
img = cv2.imread("peppers_0.04.bmp", cv2.IMREAD_GRAYSCALE)

# Apply Gaussian blur to smooth the image
img_blurred = cv2.GaussianBlur(img, (3, 3), 1)

# Apply Sobel filter
gradient_magnitude, gradient_direction = sobel_filter(img_blurred)

# Threshold the gradient magnitude to obtain the edge map
threshold_value = 50  # Adjust this value to change the threshold
edges = np.where(gradient_magnitude > threshold_value, 255, 0).astype(np.uint8)

# Display the resulting edge map
plt.imshow(edges, cmap='gray')
plt.axis('off')
plt.savefig("Q2b/Q2b_2")

# Apply Laplacian filter for edge detection
edges1 = laplacian_filter(img)

# Display the resulting edge map
plt.imshow(edges1, cmap='gray')
plt.axis('off')
plt.savefig("Q2b/Q2b_4")
