
import numpy as np 
import sklearn



def split_train_test(X, Y, test_size=0, USE_ALL=False, dtype='numpy'):
    
    assert dtype in ['numpy', 'list']
    assert (test_size or USE_ALL)
    
    print("split_train_test:")
    if dtype == 'numpy':
        print("\tData size = {}, feature dimension = {}".format(X.shape[0], X.shape[1]))
        if USE_ALL:
            tr_X = np.copy(X)
            tr_Y = np.copy(Y)
            te_X = np.copy(X)
            te_Y = np.copy(Y)
        else:
            f = sklearn.model_selection.train_test_split
            tr_X, te_X, tr_Y, te_Y = f(X, Y, test_size=test_size, random_state=14123)
    elif dtype == 'list':
        print("\tData size = {}, feature dimension = {}".format(len(X), len(X[0])))
        if USE_ALL:
            tr_X = X[:]
            tr_Y = Y[:]
            te_X = X[:]
            te_Y = Y[:]
        else:
            N = len(Y)
            train_size = int((1-test_size)*N)
            randidx = np.random.permutation(N)
            n1, n2 = randidx[0:train_size], randidx[train_size:]
            def get(arr_vals, arr_idx):
                return [arr_vals[idx] for idx in arr_idx]
            tr_X = get(X, n1)[:]
            tr_Y = get(Y, n1)[:]
            te_X = get(X, n2)[:]
            te_Y = get(Y, n2)[:]
    print("\tNum training: ", len(tr_Y))
    print("\tNum evaluation:  ", len(te_Y))
    return tr_X, tr_Y, te_X, te_Y

def split_train_eval_test(X, Y, ratios=[0.8, 0.1, 0.1], dtype='list'):
    
    X1, Y1, X2, Y2 = split_train_test(
        X, Y, 
        1-ratios[0], 
        dtype=dtype)
    
    X2, Y2, X3, Y3 = split_train_test(
        X2, Y2, 
        ratios[2]/(ratios[1]+ratios[2]),
        dtype=dtype)
    
    tr_X, tr_Y, ev_X, ev_Y, te_X, te_Y = X1, Y1, X2, Y2, X3, Y3 
    return tr_X, tr_Y, ev_X, ev_Y, te_X, te_Y