
import numpy as np 
import sklearn
from collections import OrderedDict, namedtuple
import matplotlib.pyplot as plt 
import types 

def split_train_test(X, Y, test_size=0, USE_ALL=False, dtype='numpy', if_print=True):
    
    assert dtype in ['numpy', 'list']
    
    def _print(s):
        if if_print:
            print(s)
            
    _print("split_train_test:")
    if dtype == 'numpy':
        _print("\tData size = {}, feature dimension = {}".format(X.shape[0], X.shape[1]))
        if USE_ALL:
            tr_X = np.copy(X)
            tr_Y = np.copy(Y)
            te_X = np.copy(X)
            te_Y = np.copy(Y)
        else:
            f = sklearn.model_selection.train_test_split
            tr_X, te_X, tr_Y, te_Y = f(X, Y, test_size=test_size, random_state=14123)
    elif dtype == 'list':
        _print("\tData size = {}, feature dimension = {}".format(len(X), len(X[0])))
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
    _print("\tNum training: {}".format(len(tr_Y)))
    _print("\tNum evaluation: {}".format(len(te_Y)))
    return tr_X, tr_Y, te_X, te_Y

def split_train_eval_test(X, Y, ratios=[0.8, 0.1, 0.1], dtype='list'):
    
    X1, Y1, X2, Y2 = split_train_test(
        X, Y, 
        1-ratios[0], 
        dtype=dtype, if_print=False)
    
    X2, Y2, X3, Y3 = split_train_test(
        X2, Y2, 
        ratios[2]/(ratios[1]+ratios[2]),
        dtype=dtype, if_print=False)
    
    r1, r2, r3 = 100*ratios[0], 100*ratios[1], 100*ratios[2]  
    n1, n2, n3 = len(Y1), len(Y2), len(Y3)
    print(f"Split data into [Train={n1} ({r1}%), Eval={n2} ({r2}%),  Test={n3} ({r3}%)]")
    tr_X, tr_Y, ev_X, ev_Y, te_X, te_Y = X1, Y1, X2, Y2, X3, Y3 
    return tr_X, tr_Y, ev_X, ev_Y, te_X, te_Y



class TrainingLog(object):
    def __init__(self,
                 training_args=None, # arguments in training
                #  MAX_EPOCH = 1000,
                 ):
        
        if not isinstance(training_args, dict):
            training_args = training_args.__dict__
        self.training_args = training_args 
        
        self.epochs = []
        self.accus_train = []
        self.accus_eval = []
        self.accus_test = []
        
    def store_accuracy(self, epoch, train=-0.1, eval=-0.1, test=-0.1):
        self.epochs.append(epoch)
        self.accus_train.append(train)
        self.accus_eval.append(eval)
        self.accus_test.append(test)
        # self.accu_table[epoch] = self.AccuItems(train, eval, test)
        
    def plot_train_eval_accuracy(self):
        plt.cla()
        t = self.epochs
        plt.plot(t, self.accus_train, 'r.-', label="train")
        plt.plot(t, self.accus_eval, 'b.-', label="eval")
        plt.title("Accuracy on train/eval dataset")
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        
        # lim
        # plt.ylim([0.2, 1.05])
        plt.legend(loc='upper left')
        
    def save_log(self, filename):
        with open(filename, 'w') as f:
            
            # -- Args
            f.write("Args:" + "\n")
            for key, val in self.training_args.items():
                s = "\t{:<20}: {}".format(key, val)
                f.write(s + "\n")
            f.write("\n"
                    )
            # -- Accuracies
            f.write("Accuracies:" + "\n")
            f.write("\t{:<10}{:<10}{:<10}{:<10}\n".format(
                "Epoch", "Train", "Eval", "Test"))
            
            for i in range(len(self.epochs)):
                epoch = self.epochs[i]
                train = self.accus_train[i]
                eval = self.accus_eval[i]
                test = self.accus_test[i]
                f.write("\t{:<10}{:<10.3f}{:<10.3f}{:<10.3f}\n".format(
                    epoch, train, eval, test))
                
def test_logger():
    
    # Set arguments 
    args = types.SimpleNamespace()
    args.input_size = 12  
    args.weight_decay = 0.00
    args.data_folder = "data/data_train/"

    # Test
    log = TrainingLog(training_args=args)
    log.store_accuracy(1, 0.7, 0.2)
    log.store_accuracy(5, 0.8, 0.3)
    log.store_accuracy(10, 0.9, 0.4)
    log.plot_train_eval_accuracy()
    log.save_log("tmp_from_lib_ml_test_logger.txt")
    plt.show()
    
if __name__ == "__main__":
    test_logger()