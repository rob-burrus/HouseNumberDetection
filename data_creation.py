import h5py
import numpy as np
import scipy.io
import tqdm
import cv2
import glob
import os


# getName returns the 'name' string for for the n(th) digitStruct.
def getName(file, name, n):
    return ''.join([chr(c[0]) for c in file[name[n][0]].value])

# getAllDigitStructure returns all the digitStruct from the input file.
def getAllDigitStructure(file, name):
    print('Starting get all digit structure')
    return [getName(file, name, i) for i in tqdm.tqdm(range(len(name)))]

def loadSVHN_format2():
    filename = 'svhn_data2_original/train.mat'
    data = scipy.io.loadmat(filename)
    X = data['X']
    X = np.rollaxis(X, 3)
    y = data['y']
    print(X.shape)
    print(y.shape)

def loadSVHN_format1():
    input_data1_train = 'svhn_data1_original/train/digitStruct.mat'
    file = h5py.File(input_data1_train, 'r')
    name = file['digitStruct']['name']
    names = getAllDigitStructure(file, name)
    print(names[0])
    file.close()

# loadSVHN_format2()


def loadImages():
    images = []
    filenames = glob.glob('svhn_data1_original/extra/*.png')
    filenames = filenames[:110000]
    for filename in filenames:
        img = cv2.imread(filename)
        if len(img) >= 32 and len(img[0]) >= 32:
            images.append(img)

    return images

def cropImages(images):
    i = 25933
    for img in images:
        cropped = img[0:32, 0:32, :]
        cv2.imwrite('nondigits/{}.png'.format(i), cropped)
        i += 1

# images = loadImages()
# cropImages(images)
# print(len(images))

def loadNonDigitImages():
    images = []
    filenames = glob.glob('nondigits/*.png')
    total = len(filenames)
    count = 0
    for filename in filenames:
        img = cv2.imread(filename)
        images.append(img)
        count += 1
        if count % 1000 == 0:
            print('Count: ', count)
    arr = np.stack(images, axis=0)
    return arr

def createNonDigitsMatFile(images):
    filename = 'nondigits.mat'
    y = ['10' for i in range(len(images))]
    y = np.array(y)
    y = y.reshape((len(images), 1))
    data = {
        'X': images,
        'y': y
    }
    data = scipy.io.savemat(filename, data)

def loadNonDigitsMat():
    filename = 'nondigits.mat'
    data = scipy.io.loadmat(filename)
    X = data['X']
    y = data['y']
    print(X.shape)
    print(y.shape)

# cropped = loadNonDigitImages()
# createNonDigitsMatFile(cropped)
# loadNonDigitsMat()

def loadDigitsMat():
    filename = 'svhn_data2_original/train.mat'
    data = scipy.io.loadmat(filename)
    Xtrain = data['X']
    Xtrain = np.rollaxis(Xtrain, 3)
    ytrain = data['y'].flatten()
    ytrain[ytrain == 10] = 0
    ytrain = ytrain.reshape((len(ytrain), 1))
    print(Xtrain.shape)
    print(ytrain.shape)
    filename = 'svhn_data2_original/test.mat'
    data = scipy.io.loadmat(filename)
    Xtest = data['X']
    Xtest = np.rollaxis(Xtest, 3)
    ytest = data['y']
    ytest[ytest == 10] = 0
    ytest = ytest.reshape((len(ytest), 1))
    print(Xtest.shape)
    print(ytest.shape)

    print('final')
    Xfinal = np.concatenate((Xtrain, Xtest))
    yfinal = np.concatenate((ytrain, ytest))
    print(Xfinal.shape)
    print(yfinal.shape)
    filename = 'digits.mat'
    data = {
        'X': Xfinal,
        'y': yfinal
    }
    data = scipy.io.savemat(filename, data)

# loadDigitsMat()

def combineDigitsNonDigits():
    filename = 'digits.mat'
    data = scipy.io.loadmat(filename)
    Xdigits = data['X']
    ydigits = data['y']
    print(Xdigits.shape)
    print(ydigits.shape)
    filename = 'nondigits.mat'
    data = scipy.io.loadmat(filename)
    Xnondigits = data['X']
    ynondigits = data['y']
    print(Xnondigits.shape)
    print(ynondigits.shape)

    print('final')
    Xfinal = np.concatenate((Xdigits, Xnondigits))
    yfinal = np.concatenate((ydigits, ynondigits))
    print(Xfinal.shape)
    print(yfinal.shape)
    filename = 'data.mat'
    data = {
        'X': Xfinal,
        'y': yfinal
    }
    data = scipy.io.savemat(filename, data)

# combineDigitsNonDigits()
    
