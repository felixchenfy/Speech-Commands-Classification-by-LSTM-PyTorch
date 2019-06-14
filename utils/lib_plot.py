
import numpy as np 
import matplotlib.pyplot as plt 
import soundfile as sf
import librosa
import librosa.display
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

# ----------------------------------------------------
# -- Plot audio data
# ----------------------------------------------------


def cv2_imshow(img, window_name="window name"):
    cv2.imshow(window_name, img)
    q = cv2.waitKey()
    cv2.destroyAllWindows()
    return q 

def plot_audio(data, sample_rate, yrange=(-1.1, 1.1), ax=None):
    if ax is None:
        plt.figure(figsize=(8, 5))
    t = np.arange(len(data)) / sample_rate
    plt.plot(t, data)
    plt.xlabel('time (s)')
    plt.ylabel('Intensity')
    plt.title(f'Audio with {len(data)} points, and a {sample_rate} sample rate, ')
    plt.axis([None, None, yrange[0], yrange[1]])

def plot_mfcc(mfcc, sample_rate, method='librosa', ax=None):
    if ax is None:
        plt.figure(figsize=(8, 5))
    assert method in ['librosa', 'cv2']
    
    if method == 'librosa':
        librosa.display.specshow(
            mfcc, sr=sample_rate, x_axis='time')
        plt.colorbar()
        plt.title(f'MFCCs features, len = {mfcc.shape[1]}')
        plt.tight_layout()
    
    elif method == 'cv2':
        cv2.imshow("MFCCs features", mfcc)
        q = cv2.waitKey()
        cv2.destroyAllWindows()

def plot_mfcc_histogram(
        mfcc_histogram, bins, binrange, col_divides,
    ): 
    plt.figure(figsize=(15, 5))
    # Plot by plt
    plt.imshow(mfcc_histogram)
    
    # Add notations
    plt.xlabel(f'{bins} bins, {binrange} range, and {col_divides} columns(pieces)')
    plt.ylabel("Each feature's percentage")
    plt.title("Histogram of MFCCs features")
    plt.colorbar()

# ----------------------------------------------------
# -- Plot machine learning results
# ----------------------------------------------------

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues,
                          size = None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    tmp = unique_labels(y_true, y_pred)
    # classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # print(cm)

    fig, ax = plt.subplots()
    if size is None:
        size = (12, 8)
    fig.set_size_inches(size[0], size[1])
    
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax, cm
