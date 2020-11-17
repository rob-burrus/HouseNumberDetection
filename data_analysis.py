from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import random

def loadData():
    filename = 'data.mat'
    data = loadmat(filename)
    X = data['X']
    y = data['y']
    print('Num images: ', len(X))
    print('Image shape: ', X[0].shape)
    print('Num classes: ', len(set(y.flatten())))
    return X, y


def showClassDistributions():
    X, y = loadData()
    fig, ax = plt.subplots(2, figsize=(20,15))
    fig.subplots_adjust(hspace = .3)
    labels, counts = np.unique(y, return_counts=True)
    ax[0].bar(labels, counts)
    ax[0].set_xticks(np.arange(12))
    ax[0].set_xticklabels(labels)
    ax[0].set_xlabel('Class #')
    ax[0].set_ylabel('Frequency')
    ax[0].set_title('Class Distribution')

    plt.show()

def showExampleDigitImages():
    X, y = loadData()
    fig, axs = plt.subplots(2,8, figsize=(20,5))
    fig.subplots_adjust(hspace = .2, wspace=.3)
    axs = axs.ravel()
    for i in range(0,16):
        index = random.randint(0, 100000)
        image = X[index].squeeze()
        axs[i].imshow(image)
        axs[i].axis('off')
        axs[i].set_title("Label: %d" % y[index])
    plt.show()





# showClassDistributions()
# showExampleDigitImages()