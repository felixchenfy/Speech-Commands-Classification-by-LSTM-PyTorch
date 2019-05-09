

import numpy as np 
import cv2

def change_suffix(s, new_suffix):
    i = s.rindex('.')
    s = s[:i+1] + new_suffix
    return s 

def cv2_image_f2i(img):
    img = (img*255).astype(np.uint8)
    row, col = img.shape
    rate = int(200 / img.shape[0])*1.0
    if rate >= 2:
        img = cv2.resize(img, (int(col*rate), int(row*rate)))
    return img

if __name__=="__main__":
    print(change_suffix("abc.jpg", new_suffix='avi'))