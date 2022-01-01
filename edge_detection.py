import math
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

Lap_1 = np.ones((3,3))*(-1)
Lap_1[1,1] = 8

Lap_2 = np.ones((3,3))*(-1)
Lap_2[1,1] = -4
Lap_2[0,1] = Lap_2[1,0] = Lap_2[1,2] = Lap_2[2,1] = 22

def sobel_edge_detection(img_array):
   
    Gx = np.array([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]])
    Gy = np.array([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])
    [rows, columns] = np.shape(img_array)
    result = np.zeros(shape=(rows, columns))

    
    for i in range(rows - 2):
        for j in range(columns - 2):
            gx = np.sum(np.multiply(Gx, img_array[i:i + 3, j:j + 3]))
            gy = np.sum(np.multiply(Gy, img_array[i:i + 3, j:j + 3]))
            result[i + 1, j + 1] = np.hypot(gx, gy)

    result = np.clip(result, 0, 255)
    
    return result

def create_gaussian_filter(x, y, sigma):
    gaussian_filter = np.zeros((x + x, y + x))
    total = 0
    for i in range(-x, x):
        for j in range(-y, y):
            a = (i ** 2) + (j ** 2)
            b = 2 * (sigma ** 2)
            gaussian_filter[i + 3, j + 3] = np.exp(-((a * a) / b)) / math.pi * b
            total = total + gaussian_filter[i + 3, j + 3]

    gaussian_filter = (gaussian_filter[:] / total)

    return gaussian_filter
def pad(img, padding):
    rows, cols = np.shape(img)
    padded = np.zeros((rows + padding, cols + padding))

    for i in range(rows + padding):
        for j in range(cols + padding):
            if i != 0 and i != rows + padding - 1 and j != 0 and j != cols + padding - 1:
                padded[i, j] = img[i - padding - 1, j - padding - 1]

    return padded
def apply_gaussian_filter(img, sigma):
    padded = pad(img, 4)
    rows, cols = padded.shape
    blurred = np.zeros((rows - 6, cols - 6))

    gaussian = create_gaussian_filter(3, 3, sigma)

    for i in range(rows):
        for j in range(cols):
            pixel = 0
            if i < rows - 6 and j < cols - 6:
                for k in range(6):
                    for l in range(6):
                        pixel = pixel + (gaussian[k, l] * padded[i + k, j + l])
                blurred[i, j] = int(round(pixel))

    return blurred


def LoG_edge_detection(img_array):
    sigma = 2
    kernel = Lap_1
    blurred = apply_gaussian_filter(img_array, sigma)
    blur_padded = pad(blurred, 2)
    rows, cols = blur_padded.shape
    x = len(kernel)
    y = len(kernel)

    result = np.zeros((rows - 2, cols - 2))

    for i in range(rows):
        for j in range(cols):
            pixel = 0
            if i != 0 and i < rows - 2 and j != 0 and j < cols - 2:
                for k in range(3):
                    for l in range(3):
                        pixel = pixel + (kernel[k][l] * blur_padded[i + k, j + l])
                result[i, j] = int(round(pixel))
    
    return result

def showResult(img):
    img_array = np.array(img)
    plt.figure(figsize=(10,10))
    
    plt.subplot(1,3,1)
    plt.title("Original Image")
    plt.axis('off') 
    plt.imshow(Image.fromarray(np.uint8(img)))
    
    plt.subplot(1,3,2)
    plt.title("Sobel")
    plt.axis('off')
    plt.imshow(Image.fromarray(np.uint8(sobel_edge_detection(img_array))))
    
    plt.subplot(1,3,3)
    plt.title("Laplacian of a Gaussian (LoG)")
    plt.axis('off')
    plt.imshow(Image.fromarray(np.uint8(LoG_edge_detection(img_array))))
    
    return

showResult(Image.open('HW4_test_image\image1.jpg').convert('L'))
showResult(Image.open('HW4_test_image\image2.jpg').convert('L'))
showResult(Image.open('HW4_test_image\image3.jpg').convert('L'))

plt.show()