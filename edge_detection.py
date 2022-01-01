import numpy as np
import scipy as sp
import scipy.ndimage as nd
from PIL import Image
import matplotlib.pyplot as plt

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

def any_neighbor_zero(img, i, j):
    for k in range(-1,2):
      for l in range(-1,2):
         if img[i+k, j+k] == 0:
            return True
    return False
def zero_crossing(img):
    img[img > 0] = 1
    img[img < 0] = 0
    out_img = np.zeros(img.shape)
    for i in range(1,img.shape[0]-1):
        for j in range(1,img.shape[1]-1):
            if img[i,j] > 0 and any_neighbor_zero(img, i, j):
                out_img[i,j] = 255
    return out_img

def LoG_edge_detection(img_array):
    result  = zero_crossing(nd.gaussian_laplace(img_array, sigma=3))
    
    # LoG = nd.gaussian_laplace(img_array , 2)
    # thres = np.absolute(LoG).mean() * 0.75
    # output = sp.zeros(LoG.shape)
    # w = output.shape[1]
    # h = output.shape[0]

    # for y in range(1, h - 1):
    #     for x in range(1, w - 1):
    #         patch = LoG[y-1:y+2, x-1:x+2]
    #         p = LoG[y, x]
    #         maxP = patch.max()
    #         minP = patch.min()
    #         if (p > 0):
    #             zeroCross = True if minP < 0 else False
    #         else:
    #             zeroCross = True if maxP > 0 else False
    #         if ((maxP - minP) > thres) and zeroCross:
    #             output[y, x] = 1

    # plt.imshow(output)
    # plt.show()
    return result
    # return output

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
    

    
    plt.show()
    return

showResult(Image.open('HW4_test_image\image2.jpg').convert('L'))