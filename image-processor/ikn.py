import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy import signal, ndimage
import time

### Vecorized implementation using Numpy Library ###
class DisparityMap():
    def __init__(self, numDisparities, blockSize):
        self.numDisparities = numDisparities
        self.blockSize = blockSize
    
    def census_convolution(self, center, kernel_size=(5, 5)):
        row_padding, col_padding = kernel_size[0]//2, kernel_size[1]//2
        image = cv2.copyMakeBorder(center, top=row_padding, left=col_padding, right=col_padding, bottom=row_padding, borderType=cv2.BORDER_CONSTANT, value=0)
        output = np.zeros(center.shape, dtype=np.uint8)
        r, c = center.shape
        
        bits = 0
        outputs = []
        for row in range(kernel_size[0]):
            for col in range(kernel_size[1]):                  
                output = (output << 1) | (image[row:row+r, col:col+c] >= center)
                bits += 1
                if bits%8==0 and bits!=0:
                    outputs.append(output.copy())
                    output = np.zeros(center.shape, dtype=np.uint8)
                    
        if (kernel_size[0]*kernel_size[1])%8!=0:
            outputs.append(output.copy())
        #outputs = np.array(outputs)
        return outputs

    def find_difference(self, left, right, shift_val):
        left_t = left.copy()
        right_t = right.copy()
        if len(left.shape)==2:
            r, c = left.shape
            left_t[:, :c-shift_val] = left_t[:, shift_val:]
            output = np.sum(np.unpackbits(np.bitwise_xor(left_t.reshape(r*c,1), right_t.reshape(r*c,1)), axis = 1), axis=1).reshape(r, c)
        else:
            n, r, c = left_t.shape
            left_t[:, :, :c-shift_val] = left_t[:, :, shift_val:]
            output = np.sum(np.sum(np.unpackbits(np.bitwise_xor(left_t.reshape(n*r*c,1), right_t.reshape(n*r*c,1)), axis = 1), axis=1).reshape(n, r, c), axis=0)
            
        return output
        
    def CT(self, imgL, imgR, window_size):
        imgL = imgL.astype(np.float)
        imgR = imgR.astype(np.float)
        
        if len(imgL.shape)==2:
            l, w = imgL.shape
            #find census for left and right image
            left = np.array(self.census_convolution(imgL, window_size))
            right = np.array(self.census_convolution(imgR, window_size))
        else:
            l, w, h = imgL.shape
            left = []
            right = []
            for i in range(h):
                left += self.census_convolution(imgL[:,:,i], window_size)
                right += self.census_convolution(imgR[:,:,i], window_size)
            left = np.array(left)
            right = np.array(right)
        
        #Finding error for all the numDisparities
        errors = []
        for i in range(self.numDisparities):
            errors.append(self.find_difference(left, right, i))
        
        errors = np.array(errors)
        
        disparityMap = np.zeros((l, w), dtype=np.uint8)
        mid = int(self.blockSize/2)
        for i in range(mid, l-mid):
            for j in range(mid, min(w-self.numDisparities, w-mid)):
                disparityMap[i, j] = np.argmin(errors[:,i,j])
        return disparityMap
    
    def CT_with_MBM(self, imgL, imgR):
        imgL = imgL.astype(np.float)
        imgR = imgR.astype(np.float)
        
        kernels = [(1, 61), (61, 1), (11, 11), (3, 3)]
        census_convs = []
        for kernel_size in kernels:
            
            #find census for left and right image
            if len(imgL.shape)==2:
                l, w = imgL.shape
                #find census for left and right image
                left = np.array(self.census_convolution(imgL, kernel_size))
                right = np.array(self.census_convolution(imgR, kernel_size))
            else:
                l, w, h = imgL.shape
                left = []
                right = []
                for i in range(h):
                    left += self.census_convolution(imgL[:,:,i], kernel_size)
                    right += self.census_convolution(imgR[:,:,i], kernel_size)
                left = np.array(left)
                right = np.array(right)
        
            #Finding error for all the numDisparities
            errors = []
            for i in range(self.numDisparities):
                errors.append(self.find_difference(left, right, i)/(kernel_size[0]*kernel_size[1]))
            errors = np.array(errors)
            
            census_convs.append(errors)
        
        out = np.minimum(census_convs[0], census_convs[1]) 
        for i in range(2, len(kernels)):
            out = np.multiply(out, census_convs[2])
        
        disparityMap = np.zeros((l, w), dtype=np.uint8)
        mid = int(self.blockSize/2)
        for i in range(mid, l-mid):
            for j in range(mid, min(w-self.numDisparities, w-mid)):
                disparityMap[i, j] = np.argmin(out[:,i,j])
        return disparityMap
    
    def CT_with_input_kernels(self, imgL, imgR, kernels):
        imgL = imgL.astype(np.float)
        imgR = imgR.astype(np.float)
        l, w = imgL.shape
        
        census_convs = []
        for kernel_size in kernels:
            #find census for left and right image
            left = self.census_convolution(imgL, kernel_size)
            right = self.census_convolution(imgR, kernel_size)
        
            #Finding error for all the numDisparities
            errors = []
            for i in range(self.numDisparities):
                errors.append(self.find_difference(left, right, i)/(kernel_size[0]*kernel_size[1]))
            errors = np.array(errors)
            
            census_convs.append(errors)
        
        out = census_convs[0]
        for i in range(1, len(kernels)):
            out = np.multiply(out, census_convs[i])

        disparityMap = np.zeros((l, w), dtype=np.uint8)
        mid = int(self.blockSize/2)
        for i in range(mid, l-mid):
            for j in range(mid, min(w-self.numDisparities, w-mid)):
                disparityMap[i, j] = np.argmin(out[:,i,j])
        return disparityMap
    
    #Generating Locally Consistent Disparity Map
    def LCDM(self, disparityMap, kernel):
        disparityMap = disparityMap.astype(np.int64)
        output = disparityMap.copy()
        r, c = disparityMap.shape
        row_mid, col_mid = kernel[0]//2, kernel[1]//2
        
        for i in range(2*row_mid, r-2*row_mid):
            for j in range(2*col_mid, min(c-numdisparities-col_mid, c-2*col_mid)):
                temp = disparityMap[i-row_mid:i+row_mid+1, j-col_mid:j+col_mid+1]
                val = np.bincount(temp.reshape(1, kernel[0]*kernel[1])[0]).argmax()
                output[i, j] = val
        
        return output
    

imgL = cv2.imread('./shared-memory/l-frame.png')
imgR = cv2.imread('./shared-memory/l-frame.png')
ground_truth = cv2.imread('./shared-memory/l-frame.png', 0)/4
    
#Blurring the image using Gaussian filter to remove noise
imgLg = cv2.GaussianBlur(imgL, (3,3), cv2.BORDER_DEFAULT)
imgRg = cv2.GaussianBlur(imgR, (3,3), cv2.BORDER_DEFAULT)

#Finding gradients
imgLx1 = cv2.Sobel(imgL, cv2.CV_64F, 1, 0, ksize=3)
imgLy1 = cv2.Sobel(imgL, cv2.CV_64F, 0, 1, ksize=3)

imgRx1 = cv2.Sobel(imgR, cv2.CV_64F, 1, 0, ksize=3)
imgRy1 = cv2.Sobel(imgR, cv2.CV_64F, 0, 1, ksize=3)

#Creating new multi dimensional image
imgLxy = np.zeros((imgL.shape[0], imgL.shape[1], 3), np.float64)
imgLxy[:,:,0] = imgL
imgLxy[:,:,1] = imgLx1
imgLxy[:,:,2] = imgLy1

imgRxy = np.zeros((imgR.shape[0], imgR.shape[1], 3), np.float64)
imgRxy[:,:,0] = imgR
imgRxy[:,:,1] = imgRx1
imgRxy[:,:,2] = imgRy1

#Applying algorithm
blocksize = 11
numdisparities = 64
disparity = DisparityMap(numDisparities=numdisparities, blockSize=blocksize)

t1 = time.time()
disparityMap1 = disparity.CT(imgL, imgR, (11, 11))
t2 = time.time()
print(t2 -t1)
disparityMap2 = disparity.LCDM(disparityMap1, (11, 11))

t1 = time.time()
disparityMap3 = disparity.CT_with_MBM(imgL, imgR)
t2 = time.time()
print(t2 -t1)
disparityMap4 = disparity.LCDM(disparityMap3, (11, 11))

###############################################################
##Cropping the ground truth
r, c = ground_truth.shape
ground_truth[:int(blocksize/2), :] = 0
ground_truth[:, :int(blocksize/2)] = 0
ground_truth[r-int(blocksize/2):, :] = 0
ground_truth[:, c-numdisparities:] = 0

##Calculating error
error1 = (abs(ground_truth-disparityMap1)>1) & (ground_truth!=0)
error_per1 = np.round(100*sum(error1.reshape(r*c, 1))[0]/(r*c), 2)

error2 = (abs(ground_truth-disparityMap2)>1) & (ground_truth!=0)
error_per2 = np.round(100*sum(error2.reshape(r*c, 1))[0]/(r*c), 2)

error3 = (abs(ground_truth-disparityMap3)>1) & (ground_truth!=0)
error_per3 = np.round(100*sum(error3.reshape(r*c, 1))[0]/(r*c), 2)

error4 = (abs(ground_truth-disparityMap4)>1) & (ground_truth!=0)
error_per4 = np.round(100*sum(error4.reshape(r*c, 1))[0]/(r*c), 2)

##################################################################
plt.figure(figsize=(18, 72))
plt.subplot(141)
plt.imshow(disparityMap1, 'gray')
plt.title('CT')

plt.subplot(142)
plt.imshow(error1, 'gray')
plt.title('Bad Error 1.0= '+str(error_per1)+'%')

plt.subplot(143)
plt.imshow(disparityMap2, 'gray')
plt.title('CT + LCDM')

plt.subplot(144)
plt.imshow(error2, 'gray')
plt.title('Bad Error 1.0= '+str(error_per2)+'%')
plt.show()

##################################
plt.figure(figsize=(18, 72))
plt.subplot(141)
plt.imshow(disparityMap3, 'gray')
plt.title('CT + MBM')

plt.subplot(142)
plt.imshow(error3, 'gray')
plt.title('Bad Error 1.0= '+str(error_per3)+'%')

plt.subplot(143)
plt.imshow(disparityMap4, 'gray')
plt.title('CT + MBM + LCDM')

plt.subplot(144)
plt.imshow(error4, 'gray')
plt.title('Bad Error 1.0= '+str(error_per4)+'%')
plt.show()

###################################################################
plt.figure(figsize=(18, 27))
plt.subplot(121)
plt.imshow(imgR, 'gray')
plt.title('Right Image')

plt.subplot(122)
plt.imshow(ground_truth, 'gray')
plt.title('Ground Truth')
plt.show()