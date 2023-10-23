import cv2
import numpy as np

img = cv2.imread('einstein-low-contrast.tif', cv2.IMREAD_GRAYSCALE)

def H_x(img,size):
    x_1 = img.min()
    x_k = img.max()
    height, width = img.shape
    H_x = []
    half_size = int((size-1) / 2)
    for l in range(256):
        h_x = []
        p = np.where(img == l)
        for k in range(256):
            h = 0
            w = np.abs(l - k + 1) / (x_k - x_1 + 1) 
            for i, j in zip(p[0], p[1]):
                if i >= half_size and i <= height - (half_size+1) and j >= half_size and j <= width - (half_size+1):
                    h += (img[(i- half_size):(i + half_size+1), (j - half_size):(j + half_size+1)] == k).sum()
            h_x.append(h*w )
        H_x.append(h_x)
    return H_x

def CVCE(img,size):
    height, width = img.shape
    H_x = H_x(img,size)

    Total = 0
    for i in range(256):
        for j in range(256):
            Total += H_x[i][j]

    cdf_p = []
    for i in range(256):
        pdf = 0
        for j in range(i):
            for k in range(i):
                pdf += H_x[j][k]
        cdf_p.append(pdf / Total)

    cdf_u = []
    for i in range(256):
        cdf_u.append((i + 1)**2 / (256**2))

    ind = []
    for i in range(256):
        ind.append((np.abs(np.array(cdf_p[i]) - np.array(cdf_u))).argmin())
        
    # Map intensity values using the equalized sub-histograms
    CVCE_img = np.empty((height,width))
    for i in range(256):
        cond = np.where(img == i)
        CVCE_img[cond[0], cond[1]] = ind[i]
    CVCE_img = CVCE_img.astype(np.uint8)

    return CVCE_img

cv2.imwrite("Q5/CVCE_einstein-low-contrast.tif", CVCE(img,7))