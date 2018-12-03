import subprocess
import os

import numpy as np
from sklearn.metrics import accuracy_score
from keras_contrib.layers import CRF
from keras.models import load_model

def x_and_y(data):
    tokens, segmentations = [], []
    for token in data:
        chars, labels = [], []
        for idx, char in enumerate(token):
            if char != '-':
                chars.append(char)
            else:
                continue
            if idx == 0:
                labels.append(0)
            else:
                if token[idx - 1] == '-':
                    labels.append(1) # beginning of syllable
                else:
                    labels.append(0)
        tokens.append(chars)
        segmentations.append(labels)
    return tokens, segmentations


def load_file(p, max_from_file=None):
    with open(p, 'r') as f:
        items = [l.strip() for l in f if l.strip()]
    if max_from_file:
        return items[:max_from_file]
    else:
        return items


def load_splits(input_dir, max_from_file=None):
    train = load_file(os.sep.join((input_dir, 'train.txt')), max_from_file)
    dev = load_file(os.sep.join((input_dir, 'dev.txt')), max_from_file)
    test = load_file(os.sep.join((input_dir, 'test.txt')), max_from_file)

    train = x_and_y(train)
    dev = x_and_y(dev)
    test = x_and_y(test)

    return train, dev, test

def pred_to_classes(X):
    """
    * Convert the 3-dimensional representation of class labels
      (nb_words, nb_timesteps, 3) to a 2-dimensional representation
      of shape ((nb_words, nb_timesteps)).
    """
    words = []
    for w in X:
        words.append([np.argmax(p) for p in w])
    return np.array(words)

def stringify(orig_token, segmentation, rm_symbols=True):
    """
    * Takes an original, unsyllabified `orig_token` (e.g. seruaes)
      and aligns it with a syllabification proposed (`segmentation`).
    * Returns the syllabified token in string format (e.g. ser-uaes).
    """
    orig_token = list(orig_token)

    if rm_symbols:
        # cut off <BOS> and <EOS>
        s = segmentation[1 : len(orig_token) + 1]
    else:
        s = segmentation

    new_str = []
    for p in s[::-1]:
        if p == 0:
            new_str.append(orig_token[-1])
            del orig_token[-1]
        else:
            new_str.append(orig_token[-1])
            del orig_token[-1]
            new_str.append('-')
    return ''.join(new_str[::-1])

def create_custom_objects():
    instanceHolder = {"instance": None}
    class ClassWrapper(CRF):
        def __init__(self, *args, **kwargs):
            instanceHolder["instance"] = self
            super(ClassWrapper, self).__init__(*args, **kwargs)
    def loss(*args):
        method = getattr(instanceHolder["instance"], "loss_function")
        return method(*args)
    def accuracy(*args):
        method = getattr(instanceHolder["instance"], "accuracy")
        return method(*args)
    return {"ClassWrapper": ClassWrapper ,"CRF": ClassWrapper, "loss": loss, "accuracy":accuracy}

def load_keras_model(path):
    model = load_model(path, custom_objects=create_custom_objects())
    return model