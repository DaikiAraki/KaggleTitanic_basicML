# Copyright 2022 Daiki Araki. All Rights Reserved.

import numpy as np


def calc_accuracy(y, t, by_labels=True):
    """
    calculate acculacy (0. <= accuracy <= 1.)
    :param y: model output. [batch, class] or [batch]
    :param t: target. [batch, class] or [batch]
    :param by_labels: whether the output if class labels ([batch, class])
    :return accuracy: [] (=scalar)
    """
    if by_labels:
        accuracy = np.mean(np.argmax(y, axis=1, keepdims=False) == np.argmax(t, axis=1, keepdims=False))
    else:
        accuracy = np.mean(y == t)
    return accuracy


def loss_function(y, t, by_labels=True):
    """
    MAE (Mean Absolute Error)
    :param y: estimation results
    :param t: target
    :param by_labels: whether the output if class labels ([batch, class_number])
    :return loss: [] (=scalar)
    """
    if by_labels:
        loss = np.mean(np.abs(t - y), keepdims=False)
    else:
        loss = 2 * np.mean(np.abs(t - y), keepdims=False)
    return loss


def output_to_label_indicator(x, by_labels=True):
    """
    :param x: estimation results
    :param by_labels: whether the output if class labels ([batch, class_number])
    :return y: array data for submisson
    """
    if by_labels:
        y = np.argmax(x, axis=1, keepdims=False)
    else:
        y = np.round(x).astype(np.int_)
    return y



