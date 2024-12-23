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
        return errors
    
    def CT_with_MBM(self, imgL, imgR):
        print('aaaaaaaa')
        imgL = imgL.astype(np.float)
        imgR = imgR.astype(np.float)
        
        kernels = [(1, 61), (61, 1), (11, 11), (3, 3)]
        census_convs = []
        for kernel_size in kernels:
            print('bbbbbbbbb')
            
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
        
        return out
    
    def SAD(self, imgL, imgR):
        imgL = imgL.astype(np.float)
        imgR = imgR.astype(np.float)
        difference = []
        if len(imgL.shape) == 2:
            l, w = imgL.shape
            h = 1
        else:
            l, w, h = imgL.shape
        
        for i in range(self.numDisparities):            
            temp = np.zeros(imgL.shape, dtype = np.float)
            temp[:, :w-i] = abs(imgR[:, :w-i]-imgL[:, i:])
            temp = ndimage.uniform_filter(temp, self.blockSize)
            
            for i in range(1, h):
                temp[:, :, 0] += temp[:, :, i]
            
            if len(imgL.shape)==2:
                difference.append(temp.copy())
            else:
                difference.append(temp[:, :, 0].copy())

        difference = np.array(difference)
        return difference
    
    def SAD_with_MBM(self, imgL, imgR):
        imgL = imgL.astype(np.float)/255
        imgR = imgR.astype(np.float)/255
        
        kernel_61_1 = np.ones((61, 1))
        kernel_1_61 = np.ones((1, 61))
        kernel_11_11 = np.ones((11, 11))
        kernel_3_3 = np.ones((3, 3))
        
        errors_61_1 = []
        errors_1_61 = []
        errors_11_11 = []
        errors_3_3 = []
        
        if len(imgL.shape) == 2:
            l, w = imgL.shape
            h = 1
        else:
            l, w, h = imgL.shape
        
        for i in range(self.numDisparities):
            temp = np.zeros(imgL.shape, dtype = np.float)
            temp[:, :w-i] = abs(imgR[:, :w-i]-imgL[:, i:])
            
            if len(imgL.shape)==2:
                error_61_1 = signal.convolve2d(temp, kernel_61_1, boundary='symm', mode='same')/61
                error_1_61 = signal.convolve2d(temp, kernel_1_61, boundary='symm', mode='same')/61
                error_11_11 = signal.convolve2d(temp, kernel_11_11, boundary='symm', mode='same')/121
                error_3_3 = signal.convolve2d(temp, kernel_3_3, boundary='symm', mode='same')/9
            else:
                error_61_1 = np.zeros((l, w))
                error_1_61 = np.zeros((l, w))
                error_11_11 = np.zeros((l, w))
                error_3_3 = np.zeros((l, w))
                for j in range(h):
                    error_61_1 += signal.convolve2d(temp[:, :, j], kernel_61_1, boundary='symm', mode='same')/61
                    error_1_61 += signal.convolve2d(temp[:, :, j], kernel_1_61, boundary='symm', mode='same')/61
                    error_11_11 += signal.convolve2d(temp[:, :, j], kernel_11_11, boundary='symm', mode='same')/121
                    error_3_3 += signal.convolve2d(temp[:, :, j], kernel_3_3, boundary='symm', mode='same')/9
            
            errors_61_1.append(error_61_1.copy())
            errors_1_61.append(error_1_61.copy())
            errors_11_11.append(error_11_11.copy())
            errors_3_3.append(error_3_3.copy())
        
        errors_61_1 = np.array(errors_61_1)
        errors_1_61 = np.array(errors_1_61)
        errors_11_11 = np.array(errors_11_11)
        errors_3_3 = np.array(errors_3_3)
        
        errors_61_1 = np.minimum(errors_61_1, errors_1_61)
        out = np.multiply(errors_61_1, errors_11_11)
        out = np.multiply(out, errors_3_3)
        
        return out
    
    def ProposedMethodGray(self, imgL, imgR):
        #Gray scale image
        imgL_Gray = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
        imgR_Gray = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
        
        ##############################################################################################
        #Finding gradients
        imgLx1 = cv2.Sobel(imgL_Gray, cv2.CV_64F, 1, 0, ksize=3)
        imgLy1 = cv2.Sobel(imgL_Gray, cv2.CV_64F, 0, 1, ksize=3)

        imgRx1 = cv2.Sobel(imgR_Gray, cv2.CV_64F, 1, 0, ksize=3)
        imgRy1 = cv2.Sobel(imgR_Gray, cv2.CV_64F, 0, 1, ksize=3)

        #Creating new multi dimensional image
        imgLxy = np.zeros((imgL.shape[0], imgL.shape[1], 3), np.float64)
        imgLxy[:,:,0] = imgL_Gray
        imgLxy[:,:,1] = imgLx1
        imgLxy[:,:,2] = imgLy1

        imgRxy = np.zeros((imgR.shape[0], imgR.shape[1], 3), np.float64)
        imgRxy[:,:,0] = imgR_Gray
        imgRxy[:,:,1] = imgRx1
        imgRxy[:,:,2] = imgRy1
        
        global Cc, CADc, CADg
        #Finding errors using census transform by combining color and gradients and forming a new 3d image
        Cct = self.CT_with_MBM(imgLxy, imgRxy)
        print("Finished census Transform")
        
        #Finding errors using SAD in color space
        CSADg = self.SAD_with_MBM(imgL_Gray, imgR_Gray)
        print("Finished SAD+MBM in Gray space")
        
        #Finding errors using SAD for graidents of the left and right image which give a new 3d image
        CSADgrad = self.SAD_with_MBM(imgLxy, imgRxy)
        print("Finished SAD+MBM in gradient space")
        
        #Lct, LSADg, LSADgrad = 45, 5, 18
        Lct, LSADg, LSADgrad = np.max(Cct)*20, np.max(CSADg), np.max(CSADgrad)
        C = 3 - 1/np.e**(Cct/Lct) - 1/np.e**(CSADg/LSADg) - 1/np.e**(CSADgrad/LSADgrad)
        
        l, w, h = imgL.shape
        disparityMap = np.zeros((l, w), dtype=np.float)
        mid = int(self.blockSize/2)
        for i in range(mid, l-mid):
            for j in range(mid, min(w-self.numDisparities, w-mid)):
                disparityMap[i, j] = np.argmin(C[:,i,j])
        
        return disparityMap
    
    def ProposedMethodColor(self, imgL, imgR):
        #Gray scale image
        imgL_Gray = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
        imgR_Gray = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
        
        ##############################################################################################
        #Finding gradients
        imgLx1 = cv2.Sobel(imgL_Gray, cv2.CV_64F, 1, 0, ksize=3)
        imgLy1 = cv2.Sobel(imgL_Gray, cv2.CV_64F, 0, 1, ksize=3)

        imgRx1 = cv2.Sobel(imgR_Gray, cv2.CV_64F, 1, 0, ksize=3)
        imgRy1 = cv2.Sobel(imgR_Gray, cv2.CV_64F, 0, 1, ksize=3)

        #Creating new multi dimensional image
        imgLxy = np.zeros((imgL.shape[0], imgL.shape[1], 3), np.float64)
        imgLxy[:,:,0] = imgL_Gray
        imgLxy[:,:,1] = imgLx1
        imgLxy[:,:,2] = imgLy1

        imgRxy = np.zeros((imgR.shape[0], imgR.shape[1], 3), np.float64)
        imgRxy[:,:,0] = imgR_Gray
        imgRxy[:,:,1] = imgRx1
        imgRxy[:,:,2] = imgRy1
        
        #Finding errors using census transform by combining color and gradients and forming a new 3d image
        Cct = self.CT_with_MBM(imgLxy, imgRxy)
        print("Finished census Transform")
        
        #Finding errors using SAD in color space
        CSADc = self.SAD_with_MBM(imgL, imgR)
        print("Finished SAD in color space")
        
        #############################################################################################
        #Finding gradients
        imgLx1 = cv2.Sobel(imgL, cv2.CV_64F, 1, 0, ksize=3)
        imgLy1 = cv2.Sobel(imgL, cv2.CV_64F, 0, 1, ksize=3)

        imgRx1 = cv2.Sobel(imgR, cv2.CV_64F, 1, 0, ksize=3)
        imgRy1 = cv2.Sobel(imgR, cv2.CV_64F, 0, 1, ksize=3)

        #Creating new multi dimensional image
        imgLxy = np.zeros((imgL.shape[0], imgL.shape[1], 6), np.float64)
        imgLxy[:,:,:3] = imgLx1
        imgLxy[:,:,3:] = imgLy1

        imgRxy = np.zeros((imgR.shape[0], imgR.shape[1], 6), np.float64)
        imgRxy[:,:,:3] = imgRx1
        imgRxy[:,:,3:] = imgRy1
        
        #Finding errors using SAD for graidents of the left and right image which give a new 6d image
        CSADgrad = self.SAD_with_MBM(imgLxy, imgRxy)
        print("Finished SAD in gradient space")
        
        Lct, LSADc, LSADgrad = np.max(Cct)*20, np.max(CSADc), np.max(CSADgrad)
        C = 3 - 1/np.e**(Cct/Lct) - 1/np.e**(CSADc/LSADc) - 1/np.e**(CSADgrad/LSADgrad)
        
        l, w, h = imgL.shape
        disparityMap = np.zeros((l, w), dtype=np.float)
        mid = int(self.blockSize/2)
        for i in range(mid, l-mid):
            for j in range(mid, min(w-self.numDisparities, w-mid)):
                disparityMap[i, j] = np.argmin(C[:,i,j])
        
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
    
def plot(ground_truth, disparityMap, name):
    ###############################################################
    ##Cropping the ground truth
    r, c = ground_truth.shape
    ground_truth[:int(blocksize/2), :] = 0
    ground_truth[:, :int(blocksize/2)] = 0
    ground_truth[r-int(blocksize/2):, :] = 0
    ground_truth[:, c-numdisparities:] = 0

    ##Calculating error
    error = (abs(ground_truth-disparityMap)>1) & (ground_truth!=0)
    error_per = np.round(100*sum(error.reshape(r*c, 1))[0]/(r*c), 2)
    
    ##################################################################
    plt.figure(figsize=(18, 54))
    plt.subplot(131)
    plt.imshow(ground_truth, 'gray')
    plt.title('Ground Truth')

    plt.subplot(132)
    plt.imshow(disparityMap, 'gray')
    plt.title(name)

    plt.subplot(133)
    plt.imshow(error, 'gray')
    plt.title(name+". Bad Error 1.0= "+str(error_per)+'%')
    plt.show()
    
    #Input data

# imgL = cv2.imread('tsukuba_l.png')
# imgR = cv2.imread('tsukuba_r.png')
# ground_truth = cv2.imread('tsukuba_r.png', 0)

imgL = cv2.imread('./shared-memory/im0.ppm')
imgR = cv2.imread('./shared-memory/im8.ppm')
ground_truth = cv2.imread('./shared-memory/l-frame.png', 0)/4

# imgL = cv2.imread('2001/teddy/im2.png')
# imgR = cv2.imread('2001/teddy/im6.png')
# ground_truth = cv2.imread('2001/teddy/disp6.png', 0)/4

# imgL = cv2.imread('2001/sawtooth/im2.ppm')
# imgR = cv2.imread('2001/sawtooth/im6.ppm')
# ground_truth = cv2.imread('2001/sawtooth/disp6.pgm', 0)/8

# imgL = cv2.imread('2001/venus/im2.ppm')
# imgR = cv2.imread('2001/venus/im6.ppm')
# ground_truth = cv2.imread('2001/venus/disp6.pgm', 0)/8

# imgL = cv2.imread('2001/barn1/im2.ppm')
# imgR = cv2.imread('2001/barn1/im6.ppm')
# ground_truth = cv2.imread('2001/barn1/disp6.pgm', 0)/8

# imgL = cv2.imread('2001/bull/im2.ppm')
# imgR = cv2.imread('2001/bull/im6.ppm')
# ground_truth = cv2.imread('2001/bull/disp6.pgm', 0)/8

#Applying algorithm
blocksize = 11
numdisparities = 64
disparity = DisparityMap(numDisparities=numdisparities, blockSize=blocksize)

t1 = time.time()
# disparityMap1 = disparity.SAD_with_MBM(imgL, imgR)
disparityMap1 = disparity.ProposedMethodColor(imgL, imgR)
# disparityMap2 = disparity.LCDM(disparityMap1, (11, 11))
t2 = time.time()
print(t2 -t1)

# plot(ground_truth, disparityMap1, "Total Matching Cost")
plot(ground_truth, disparityMap1, "LCDM")