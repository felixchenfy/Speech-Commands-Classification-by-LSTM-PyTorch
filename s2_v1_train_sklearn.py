
import numpy as np 
import time
from mylib.mylib_sklearn import *
from mylib.mylib_plot import *
from mylib.mylib_io import *

train_X = read_list('train_X.csv')
train_Y = read_list('train_Y.csv')
train_X, train_Y = np.array(train_X), np.array(train_Y).astype(np.int)
classes = read_list("classes.csv")
tr_X, te_X, tr_Y, te_Y = split_data(train_X, train_Y, USE_ALL=False)

# Train
model = ClassifierOfflineTrain()
model.train(tr_X, tr_Y)

# Evaluate performance

t0 = time.time()

tr_accu, tr_Y_predict = model.predict_and_evaluate(tr_X, tr_Y)
te_accu, te_Y_predict = model.predict_and_evaluate(te_X, te_Y)

print( "Time cost for predicting one sample:", (time.time() - t0) / len(train_Y) )

# Plot result ---------------------------------------------
if 1:
    axis, cf = plot_confusion_matrix(te_Y, te_Y_predict, classes, normalize=True, size=(10, 6))
    plt.show()

# Save model ---------------------------------------------
if model.model_name=="Neural Net":
    
    # Save trained model to file
    import pickle
    path_to_save_model = './models/trained_classifier.pickle'
    
    with open(path_to_save_model, 'wb') as f:
        pickle.dump(model, f)

    # Load and test again to ensure correctly saved to file
    if 1: 
        with open(path_to_save_model, 'rb') as f:
            model2 = pickle.load(f)
        print(tr_X.shape)
        model2.predict_and_evaluate(tr_X, tr_Y)
        model2.predict_and_evaluate(te_X, te_Y)