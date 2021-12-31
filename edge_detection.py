import numpy as np
import scipy.ndimage.filters
from PIL import Image
import matplotlib.pyplot as plt

def sobel_edge_detection(img_array):
    # matrix = np.asarray([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
    # result = scipy.ndimage.filters.convolve(img_array, matrix)
   
    Gx = np.array([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]])
    Gy = np.array([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])
    [rows, columns] = np.shape(img_array)
    sobel_filtered_image = np.zeros(shape=(rows, columns))


    for i in range(rows - 2):
        for j in range(columns - 2):
            gx = np.sum(np.multiply(Gx, img_array[i:i + 3, j:j + 3]))  # x direction
            gy = np.sum(np.multiply(Gy, img_array[i:i + 3, j:j + 3]))  # y direction
            sobel_filtered_image[i + 1, j + 1] = np.sqrt(gx ** 2 + gy ** 2)  # calculate the "hypotenuse"

    result = np.clip(sobel_filtered_image, 0, 255)
    
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
    plt.imshow(Image.fromarray(np.uint8(img_array)))
    

    
    plt.show()
    return

showResult(Image.open('HW4_test_image\image2.jpg').convert('L'))