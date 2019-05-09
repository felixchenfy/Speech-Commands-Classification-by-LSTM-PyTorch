
import numpy as np 
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.decomposition import PCA


# Classifier for training in jupyter notebook-----------------------------------------------
class ClassifierOfflineTrain(object):
    def __init__(self):
        self.init_all_models()
        # self.clf = self.choose_model("Nearest Neighbors")
        # self.clf = self.choose_model("Linear SVM")
        # self.clf = self.choose_model("RBF SVM") # 0.65
        # self.clf = self.choose_model("Gaussian Process")
        # self.clf = self.choose_model("Decision Tree")
        # self.clf = self.choose_model("Random Forest")
        self.clf = self.choose_model("Neural Net") # 0.6

    def choose_model(self, name):
        self.model_name = name
        idx = self.names.index(name)
        return self.classifiers[idx]
            
    def init_all_models(self):
        self.names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
             "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
             "Naive Bayes", "QDA"]
        self.model_name = None
        self.classifiers = [
            KNeighborsClassifier(5),
            SVC(kernel="linear", C=10.0),
            SVC(gamma=0.01, C=10.0, verbose=True),
            GaussianProcessClassifier(1.0 * RBF(1.0)),
            DecisionTreeClassifier(max_depth=5),
            RandomForestClassifier(max_depth=30, n_estimators=100, max_features="auto"),
            MLPClassifier((100, 100, 100, 100)), # Neural Net
            AdaBoostClassifier(),
            GaussianNB(),
            QuadraticDiscriminantAnalysis()]
        self.USE_PCA = False

    def train(self, X, Y):
        print("\n" + "-"*80 + "\nStart training:\n")

        # Apply PCA
        if self.USE_PCA:
            NUM_FEATURES_FROM_PCA = 80
            n_components = min(NUM_FEATURES_FROM_PCA, X.shape[1])
            self.pca = PCA(n_components=n_components, whiten=True)
            self.pca.fit(X)
            X = self.pca.transform(X)

            print("Apply PCA: Sum eig values = {}, X_new.shape = {}\n".format(
                np.sum(self.pca.explained_variance_ratio_), X.shape))

        # Train
        timer = Timer()
        self.clf.fit(X, Y)
        timer.report("Time cost of training = ")

        # Return
        print("\nTraining ends.\n" + "-"*80 + "\n")
        return None

    def predict(self, X):
        if len(X.shape)==1:
            X = [X]
        if self.USE_PCA:
            X = self.pca.transform(X)
        Y_predict = self.clf.predict(X)
        return Y_predict

    def predict_proba(self, X): # only works for neural network
        if self.USE_PCA:
            X = self.pca.transform(X)
        Y_probs = self.clf.predict_proba(X)
        return Y_probs

    def predict_and_evaluate(self, te_X, te_Y):
        te_Y_predict = self.predict(te_X)
        N = len(te_Y)
        n = sum( te_Y_predict == te_Y )
        accu = n / N
        print("Accuracy is ", accu)
        # print(te_Y_predict)
        return accu, te_Y_predict


def split_data(X, Y, USE_ALL=False):
    from sklearn.model_selection import train_test_split
    if USE_ALL:
        tr_X = np.copy(X)
        tr_Y = np.copy(Y)
        te_X = np.copy(X)
        te_Y = np.copy(Y)
    else:
        tr_X, te_X, tr_Y, te_Y = train_test_split(X, Y, test_size=0.3, random_state=14123)

    print("Data size = {}, feature dimension = {}".format(X.shape[0], X.shape[1]))
    print("Num training: ", len(tr_Y))
    print("Num testing:  ", len(te_Y))
    return tr_X, te_X, tr_Y, te_Y


class Timer(object):
    def __init__(self):
        self.t0 = time.time()

    def report(self, s):
        t = time.time() - self.t0
        print(s + "{:.1f}".format(t) + " seconds")

    def reset(self):
        self.t0 = time.time()
