import cv2
import numpy as np
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt

def preprocess():
    filename = 'data.mat'
    data = loadmat(filename)
    images = data['X']
    
    length = len(images)
    processed = np.empty([length, 32, 32, 3])
    for i in range(length):
        # img = cv2.cvtColor(images[i], cv2.COLOR_RGB2GRAY) #grayscale
        processed[i] = cv2.equalizeHist(images[i]) # histogram equalization
        
            
    processed = (processed - 127.5)/127.5 # normalize [-1, 1]
    
    filename = 'preprocessed.mat'
    data = {
        'X': images,
        'y': data['y']
    }
    savemat(filename, data)

def preprocessSingle(img):
    # result = cv2.equalizeHist(img) # histogram equalization
    result = (img - 127.5)/127.5 # normalize [-1, 1]
    return result

def visualizePreprocessing():
    filename = 'data.mat'
    data = loadmat(filename)
    images = data['X']
    labels = data['y']
    
    fig, axes = plt.subplots(nrows=1, ncols=6, figsize=(15,10))
    fig.subplots_adjust(hspace = .25, wspace=.3)
    axs = axes.ravel()
    length = 99289
    for i in range(0,5,2):
        index = np.random.randint(0,length)
        img = images[index]
        new_img = preprocessSingle(img)
        axs[i].imshow(img)
        axs[i].axis('off')
        axs[i].set_title("Class %d Original" % labels[index])
        axs[i+1].imshow(new_img)
        axs[i+1].axis('off')
        axs[i+1].set_title("Class %d w/ preprocessing" % labels[index]) 
    plt.show()
    
# visualizePreprocessing()
# preprocess()