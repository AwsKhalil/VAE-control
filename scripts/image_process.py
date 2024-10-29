

import cv2
import numpy as np

class ImageProcess:

    def process(self, img, bgr=True):
        return self._normalize_0_255(img, bgr)
    
    def norm_0_1(self, img):
        return self._normalize_0_1(img)
    
    def norm_n1_1(self, img):
        return self._normalize_n1_1(img)
    
    def un_norm_n1_1(self, x_norm, x_max, x_min):
        return self._un_normalize_n1_1(x_norm, x_max, x_min)
    
    def from_n1_1_to_0_1(self, img):
        return self._from_n1_1_to_0_1(img)
    
    def norm_min_max(self, arr, min_min, arr_max):
        return self._norm_min_max(arr, min_min, arr_max)
        
    # img is expected as BGR         
    def _equalize_histogram(self, img, bgr = True):

        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        # equalize the histogram of the Y channel
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        if (bgr == True):
            img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        else:
            img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
        return img
    
    def _normalize_0_255(self, img, bgr = True):

        img_norm = np.zeros_like(img)
        if (bgr != True): # if not bgr then assume it as RGB
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        cv2.normalize(img, img_norm, 0, 255, cv2.NORM_MINMAX)
        
        # img_norm = np.round((img + 1) * 255 / 2)
        
        return img_norm
    
    def _normalize_0_1(self, img, bgr = True):

        img_norm = np.zeros_like(img)
        if (bgr != True): # if not bgr then assume it as RGB
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        cv2.normalize(img, img_norm, 0, 1, cv2.NORM_MINMAX)
        
        # img_norm = np.round((img + 1) * 255 / 2)
        
        return img_norm
    
    def _normalize_n1_1(self, arr): # scale from [0,255] to [-1,1]

        arr_np = np.array(arr)
        arr_max = np.max(arr_np)
        arr_min = np.min(arr_np)
        
        arr_range = arr_max - arr_min
        arr_norm = 2*((arr_np - arr_min) / arr_range)-1
        
        # img_norm = (img - 127.5) / 127.5

        return arr_norm, arr_max, arr_min
    
    def _un_normalize_n1_1(self, x_norm, x_max, x_min): # scale from [0,255] to [-1,1]

        # arr_np = np.array(arr)
        # arr_max = np.max(arr_np)
        # arr_min = np.min(arr_np)
        
        arr_range = x_max - x_min
        arr_orig = arr_range*((x_norm +1)/2)+x_min
        
        # img_norm = (img - 127.5) / 127.5

        return arr_orig
    
    def _norm_min_max(self, x_norm, x_max, x_min):
        
        arr_range = x_max - x_min
        arr_scaled = ((x_norm - x_min) / arr_range)
        return arr_scaled
            
    def _from_n1_1_to_0_1(self, img, bgr = True): # scale from [0,255] to [-1,1]
        # img_norm = np.zeros_like(img)
        # if (bgr != True): # if not bgr then assume it as RGB
        #     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # cv2.normalize(img, img_norm, -1, 1, cv2.NORM_MINMAX)
    	
        img = (img + 1.0) / 2.0
        
        return img
