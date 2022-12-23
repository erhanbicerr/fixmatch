import collections
import gzip
import os
import tarfile
import tempfile
from urllib import request

import numpy as np
import scipy.io
import tensorflow as tf
from absl import app
from tqdm import trange

from skimage.io import imread

CAFE_PATH = "ML_DATA/CAFE_5emotions_augmented_format"



URLS = {
    'svhn': 'http://ufldl.stanford.edu/housenumbers/{}_32x32.mat',
    'cifar10': 'https://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz',
    'cifar100': 'https://www.cs.toronto.edu/~kriz/cifar-100-matlab.tar.gz',
    'stl10': 'http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz',
}


def _encode_png(images):
    raw = []
    with tf.Session() as sess, tf.device('cpu:0'):
        image_x = tf.placeholder(tf.uint8, [None, None, None], 'image_x')
        to_png = tf.image.encode_png(image_x)
        for x in trange(images.shape[0], desc='PNG Encoding', leave=False):
            raw.append(sess.run(to_png, feed_dict={image_x: images[x]}))
    return raw



def _load_cifar10():
    def unflatten(images):
        # incoming data is flattened!!!
        # incoming data has shape (n,3072->32*32*3)
        return np.transpose(images.reshape((images.shape[0], 3, 32, 32)),
                            [0, 2, 3, 1]) # n,32,32,3

    with tempfile.NamedTemporaryFile() as f:
        request.urlretrieve(URLS['cifar10'], f.name)
        tar = tarfile.open(fileobj=f)
        train_data_batches, train_data_labels = [], []
        for batch in range(1, 6):
            data_dict = scipy.io.loadmat(tar.extractfile(
                'cifar-10-batches-mat/data_batch_{}.mat'.format(batch)))
            print(len(data_dict['data']))
            print(data_dict['data'].shape)
            print(data_dict['labels'].shape)
            train_data_batches.append(data_dict['data'])
            train_data_labels.append(data_dict['labels'].flatten())
        print(len(train_data_batches))
        train_set = {'images': np.concatenate(train_data_batches, axis=0),
                     'labels': np.concatenate(train_data_labels, axis=0)}
        print(train_set["images"].shape)
        print(train_set["labels"].shape)
        data_dict = scipy.io.loadmat(tar.extractfile(
            'cifar-10-batches-mat/test_batch.mat'))
        test_set = {'images': data_dict['data'],
                    'labels': data_dict['labels'].flatten()}
    print(test_set["images"].shape)
    print(test_set["labels"].shape)
    train_set['images'] = _encode_png(unflatten(train_set['images']))
    test_set['images'] = _encode_png(unflatten(test_set['images']))
    return dict(train=train_set, test=test_set)

def _load_cafe():
    train_data_batches, train_data_labels = [], []
    
    for fold in range(1, 3):
        folder_name = f"fold{fold}_images"
        folder_path = os.path.join(CAFE_PATH, folder_name)
        # get the list of all fold images and append to train data batches
        train_data_batches.append([imread(os.path.join(folder_path,im)) for im in os.listdir(folder_path)])

        # get the labels
        label_path = os.path.join(CAFE_PATH,f"part_{fold}_label_array.npy")
        train_data_labels.append(np.load(label_path))
    train_set = {'images': np.concatenate(train_data_batches, axis=0),
                'labels': np.concatenate(train_data_labels, axis=0)}
    print(train_set["images"].shape)
    print(train_set["labels"].shape)
    # 3rd fold is the testing fold
    test_path = os.path.join(CAFE_PATH, f"fold{3}_images")
    test_label_path = os.path.join(CAFE_PATH,f"part_{3}_label_array.npy")
    # get the test images
    test_set_images = [imread(os.path.join(test_path,im)) for im in os.listdir(test_path)]
    # get the test labels
    test_data_labels = np.load(test_label_path)
    test_set = {'images': np.array(test_set_images),
                'labels': test_data_labels}
    print(test_set["images"].shape)
    print(test_set["labels"].shape)
    # return dict of train and test sets.
    train_set['images'] = _encode_png(train_set['images'])
    test_set['images'] = _encode_png(test_set['images'])
    return dict(train=train_set, test=test_set)




def main(argv):
    #figar= _load_cifar10()
    #print(np.array(figar["train"]["images"]).shape) # 50000
    #print(np.array(figar["train"]["labels"]).shape) # 50000 (10000*5 batch)

    #cafe = _load_cafe()
    #cifar_test = _load_cifar10()
    "SHAPE TESTS"
    #print(np.array(cifar_test["train"]["images"]).shape) # (50000,)
    #print(np.array(cifar_test["train"]["labels"]).shape) # (50000,)

    #print(np.array(cifar_test["test"]["images"]).shape) # (10000,)
    #print(np.array(cifar_test["test"]["labels"]).shape) # (10000,)

    "SHAPE TESTS"
    #print(np.array(cafe["train"]["images"]).shape) # (50000,)
    #print(np.array(cafe["train"]["labels"]).shape) # (50000,)

    #print(np.array(cafe["test"]["images"]).shape) # (10000,)
    #print(np.array(cafe["test"]["labels"]).shape) # (10000,)
    arr = np.load("/home/erhan/googleFixMatch/fixmatch/ML_DATA/CAFE_5emotions_augmented_format/part_1_label_array_emotion.npy")
    print(np.unique(arr,return_counts=True))
    #print(np.array(cifar_test["train"]["images"][0]))
if __name__ == '__main__':
    app.run(main)
