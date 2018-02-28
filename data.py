import numpy as np
import os
from scipy.io import loadmat
import scipy
import sys

def u2t(x):
    return x.astype('float32') / 255 * 2 - 1

def s2t(x):
    return x * 2 - 1

class Data(object):
    def __init__(self, images, labels, cast=False):
        self.images = images
        self.labels = labels
        self.cast = cast

    def next_batch(self, bs):
        idx = np.random.choice(len(self.images), bs, replace=False)
        if self.cast:
            return u2t(self.images[idx]), self.labels[idx].astype('float32')
        else:
            return self.images[idx], self.labels[idx]

class Svhn(object):
    def __init__(self, size=32):
        print "Loading SVHN"
        sys.stdout.flush()
        path = '/mnt/ilcompf5d0/user/rshu/data'
        train = loadmat(os.path.join(path, 'extra_32x32.mat'))
        test = loadmat(os.path.join(path, 'test_32x32.mat'))

        # Change format
        trainx, trainy = self.change_format(train, size)
        testx, testy = self.change_format(test, size)

        # Convert to one-hot
        trainy = np.eye(10)[trainy]
        testy = np.eye(10)[testy]

        self.train = Data(trainx, trainy, cast=True)
        self.test = Data(testx, testy, cast=True)

    @staticmethod
    def change_format(mat, size):
        x = mat['X'].transpose((3, 0, 1, 2))
        y = mat['y'].reshape(-1)
        y[y == 10] = 0
        return x, y

class Mnist(object):
    def __init__(self, size=32):
        print "Loading MNIST"
        sys.stdout.flush()
        path = '/mnt/ilcompf5d0/user/rshu/data'
        data = np.load(os.path.join(path, 'mnist.npz'))
        trainx = np.concatenate((data['x_train'], data['x_valid']), axis=0)
        trainy = np.concatenate((data['y_train'], data['y_valid']))
        trainy = np.eye(10)[trainy].astype('float32')

        testx = data['x_test']
        testy = data['y_test'].astype('int')
        testy = np.eye(10)[testy].astype('float32')

        if size == 32:
            print "Resizing MNIST"
            sys.stdout.flush()
            trainx = self.resize(trainx)
            testx = self.resize(testx)

        self.train = Data(trainx, trainy)
        self.test = Data(testx, testy)

    @staticmethod
    def resize(x):
        x = x.reshape(-1, 28, 28)
        resized_x = np.empty((len(x), 32, 32), dtype='float32')
        for i, img in enumerate(x):
            resized_x[i] = u2t(scipy.misc.imresize(img, (32, 32)))
        resized_x = resized_x.reshape(-1, 32, 32, 1)
        return resized_x
