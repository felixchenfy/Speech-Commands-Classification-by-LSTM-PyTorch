

import numpy as np 
import cv2

def get_filename(path):
    return path.split('/')[-1]

def change_suffix(s, new_suffix, index=None):
    i = s.rindex('.')
    si = ""
    if index is not None:
        si = "_" + str(index)
    s = s[:i] + si + "." + new_suffix
    return s 


def int2str(num, len):
    return ("{:0"+str(len)+"d}").format(num)

def add_idx_suffix(s, idx):
    i = s.rindex('.')
    s = s[:i] + "_" + str(idx) + s[i:]
    return s 

def cv2_image_f2i(img):
    img = (img*255).astype(np.uint8)
    row, col = img.shape
    rate = int(200 / img.shape[0])*1.0
    if rate >= 2:
        img = cv2.resize(img, (int(col*rate), int(row*rate)))
    return img


def split_data(X, Y, eval_size=0.3, USE_ALL=False, dtype='numpy'):
    
    assert dtype in ['numpy', 'list']

    if dtype == 'numpy':
        print("Data size = {}, feature dimension = {}".format(X.shape[0], X.shape[1]))
        from sklearn.model_selection import train_test_split
        if USE_ALL:
            tr_X = np.copy(X)
            tr_Y = np.copy(Y)
            te_X = np.copy(X)
            te_Y = np.copy(Y)
        else:
            tr_X, te_X, tr_Y, te_Y = train_test_split(X, Y, test_size=eval_size, random_state=14123)
    elif dtype == 'list':
        print("Data size = {}, feature dimension = {}".format(len(X), len(X[0])))
        if USE_ALL:
            tr_X = X[:]
            tr_Y = Y[:]
            te_X = X[:]
            te_Y = Y[:]
        else:
            N = len(Y)
            train_size = int((1-eval_size)*N)
            randidx = np.random.permutation(N)
            n1, n2 = randidx[0:train_size], randidx[train_size:]
            def get(arr_vals, arr_idx):
                return [arr_vals[idx] for idx in arr_idx]
            tr_X = get(X, n1)[:]
            tr_Y = get(Y, n1)[:]
            te_X = get(X, n2)[:]
            te_Y = get(Y, n2)[:]
    print("Num training: ", len(tr_Y))
    print("Num evaluation:  ", len(te_Y))
    return tr_X, te_X, tr_Y, te_Y

if __name__=="__main__":
    print(change_suffix("abc.jpg", new_suffix='avi'))