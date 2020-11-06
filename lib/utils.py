import os
import gzip
import pickle

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def tup_distance(node1, node2, mode="Euclidean"):
    """
    mode: "Manhattan", "Euclidean"
    """
    if mode=="Euclidean":
        return ((node1[0]-node2[0])**2+(node1[1]-node2[1])**2)**0.5
    elif mode=="Manhattan":
        return np.abs(node1[0]-node2[0])+np.abs(node1[1]-node2[1])
    else:
        raise ValueError("Unrecognized distance mode: "+mode)

def mat_distance(mat1, mat2, mode="Euclidean"):
    """
    mode: "Manhattan", "Euclidean"
    """
    if mode=="Euclidean":
        return np.sum((mat1-mat2)**2, axis=-1)**0.5
    elif mode=="Manhattan":
        return np.sum(np.abs(mat1-mat2), axis=-1)
    else:
        raise ValueError("Unrecognized distance mode: "+mode)
        
def to_alpha(x):
    return np.clip(x[..., 3:4], 0, 0.9999)

def to_rgb(x):
    # assume rgb premultiplied by alpha
    rgb, a = x[..., :3], to_alpha(x)
    return np.clip(1.0-a+rgb, 0, 0.9999)

def get_sobel(size):
    assert (size+1)%2==0
    gx = np.zeros([size, size])
    gy = np.zeros([size, size])
    mid = size//2
    
    for row in range(size):
        for col in range(size):
            i = col-mid
            j = row-mid
            gx[row, col] = i / max((i*i + j*j), 1e-6)
            gy[row, col] = j / max((i*i + j*j), 1e-6)
    
    return gx, gy

class SamplePool:

    def __init__(self, *, _parent=None, _parent_idx=None, **slots):
        self._parent = _parent
        self._parent_idx = _parent_idx
        self._slot_names = slots.keys()
        self._size = None
        for k, v in slots.items():
            if self._size is None:
                self._size = len(v)
            assert self._size == len(v)
            setattr(self, k, np.asarray(v))

    def sample(self, n):
        idx = np.random.choice(self._size, n, False)
        batch = {k: getattr(self, k)[idx] for k in self._slot_names}
        batch = SamplePool(**batch, _parent=self, _parent_idx=idx)
        return batch

    def commit(self):
        for k in self._slot_names:
            getattr(self._parent, k)[self._parent_idx] = getattr(self, k)

def get_living_mask(x, alpha_channel, kernel_size=3):
    return F.max_pool2d(x[:, alpha_channel:(alpha_channel+1), :, :],
                        kernel_size=kernel_size, stride=1, padding=(kernel_size//2)) > 0.1

def get_rand_avail(alive_map):
    a = np.where(alive_map>0)
    index = np.random.randint(len(a[0]))
    return (a[1][index], a[2][index])

def make_seed(shape, n_channels, alpha_channels, coord=None):
    if coord is None:
        coord = (shape[0]//2, shape[1]//2)
    seed = np.zeros([shape[0], shape[1], n_channels], np.float32)
    seed[coord[0], coord[1], alpha_channels] = 1.0
    return seed

def make_circle_masks(n, h, w, rmin=0.2, rmax=0.4):
    x = np.linspace(-1.0, 1.0, w)[None, None, :]
    y = np.linspace(-1.0, 1.0, h)[None, :, None]
    center = np.random.random([2, n, 1, 1])*1.0-0.5
    r = np.random.random([n, 1, 1])*(rmax-rmin)+rmin
    x, y = (x-center[0])/r, (y-center[1])/r
    mask = (x*x+y*y < 1.0).astype(np.float32)
    return mask

def softmax(x, axis):
    norm = np.exp(x-np.max(x, axis, keepdims=True))
    y = norm/np.sum(norm, axis, keepdims=True)
    return y

def random_aug(img, isColor=True, isTilt=True, isBlur=False,
               verbose=False, eps=1e-8):
    """
    img as float from 0 to 1
    """
    result = img
    
    if isColor:
        indicator1 = np.random.random()*2
        if indicator1>1:
            # contrast
            contrast = np.random.random()*0.8+0.6
            if verbose: print("contrast:", contrast)
            result = np.clip(img*contrast, eps, 1-eps)
        elif indicator1<1:
            # brightness
            brightness = np.random.random()*0.8-0.4
            if verbose: print("brightness:", brightness)
            result = np.clip(img+brightness, eps, 1-eps)
        
    if isTilt:
        (h, w) = result.shape[:2]
        center = (w // 2, h // 2)
        deg = np.random.randint(61)-30
        M = cv2.getRotationMatrix2D(center, deg, 1.0)
        result = cv2.warpAffine(result, M, (w, h))
        if verbose: print("rotation:", deg, "degree")
    
    if isBlur:
        indicator2 = np.random.random()*3
        if indicator2>2:
            # blurriness
            blurriness = int(np.random.randint(3))*2+3
            if verbose: print("blurriness:", blurriness)
            result = cv2.GaussianBlur(result,(blurriness,blurriness),0)
        elif indicator2<1:
            # resolution
            l = np.random.randint(12)+16
            if verbose: print("image resized to ("+str(l)+","+str(l)+")")
            original_shape = img.shape[:2]
            result = cv2.resize(result, (l,l))
            result = cv2.resize(result, original_shape)
    return np.clip(result, eps, 1-eps)

def mnist(path, isOneHot=True):
    files = ['train-images-idx3-ubyte.gz',
             'train-labels-idx1-ubyte.gz',
             't10k-images-idx3-ubyte.gz',
             't10k-labels-idx1-ubyte.gz']

    def _images(path):
        """Return images loaded locally."""
        with gzip.open(path) as f:
            # First 16 bytes are magic_number, n_imgs, n_rows, n_cols
            pixels = np.frombuffer(f.read(), 'B', offset=16)
        return pixels.reshape(-1, 784).astype('float32') / 255

    def _labels(path):
        """Return labels loaded locally."""
        with gzip.open(path) as f:
            # First 8 bytes are magic_number, n_labels
            integer_labels = np.frombuffer(f.read(), 'B', offset=8)

        def _onehot(integer_labels):
            """Return matrix whose rows are onehot encodings of integers."""
            n_rows = len(integer_labels)
            n_cols = integer_labels.max() + 1
            onehot = np.zeros((n_rows, n_cols), dtype='uint8')
            onehot[np.arange(n_rows), integer_labels] = 1
            return onehot

        return _onehot(integer_labels)

    train_images = _images(os.path.join(path, files[0]))
    train_labels = _labels(os.path.join(path, files[1]))
    test_images = _images(os.path.join(path, files[2]))
    test_labels = _labels(os.path.join(path, files[3]))
    
    if not isOneHot:
        train_labels = np.argmax(train_labels, -1)
        test_labels = np.argmax(test_labels, -1)

    return train_images, train_labels, test_images, test_labels

def cifar10(path):

    def load_CIFAR_batch(filename):
        """ load single batch of cifar """
        with open(filename, 'rb') as f:
            datadict = pickle.load(f, encoding='iso-8859-1')
            X = datadict['data']
            Y = datadict['labels']
            X = X.reshape(10000,3072)
            Y = np.array(Y)
            return X, Y

    def load_CIFAR10(ROOT):
        """ load all of cifar """
        xs = []
        ys = []
        for b in range(1,6):
            f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
            X, Y = load_CIFAR_batch(f)
            xs.append(X)
            ys.append(Y)
        Xtr = np.concatenate(xs)
        Ytr = np.concatenate(ys)
        del X, Y
        Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
        return Xtr, Ytr, Xte, Yte

    cifar10_dir = '../input/cifar-10-batches-py/'
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    X_train, y_train, X_test, y_test = load_CIFAR10(path)
    
    trains = {}
    tests = {}
    trains['data'] = X_train
    trains['labels'] = y_train
    trains['label_names'] = classes
    tests['data'] = X_test
    tests['labels'] = y_test
    tests['label_names'] = classes
    
    return trains, tests

def cifar100(path):
    files = ['train', 'test']
    
    coarse_names = ['aquatic mammals', 'fish', 'flowers', 'food_containers', 'fruit_and_vegetables',
                    'household_electrical_devices', 'household_furniture', 'insects', 'large_carnivores',
                    'large_man-made_outdoor_things', 'large_natural_outdoor_scenes',
                    'large_omnivores_and_herbivores', 'medium-sized_mammals', 'non-insect_invertebrates',
                    'people', 'reptiles', 'small_mammals', 'trees', 'vehicles_1', 'vehicles_2']

    fine_names = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 
                  'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 
                  'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 
                  'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 
                  'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 
                  'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
                  'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
                  'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
                  'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
                  'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
                  'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
                  'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
                  'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
                  'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
                  'worm']
    
    with open(os.path.join(path, files[0]), 'rb') as fo:
        trains = pickle.load(fo, encoding='iso-8859-1')
            
    with open(os.path.join(path, files[1]), 'rb') as fo:
        tests = pickle.load(fo, encoding='iso-8859-1')
            
    trains['coarse_names'] = coarse_names
    trains['fine_names'] = fine_names
    tests['coarse_names'] = coarse_names
    tests['fine_names'] = fine_names

    return trains, tests
