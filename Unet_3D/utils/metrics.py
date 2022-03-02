from functools import partial
import numpy as np
from keras import backend as K


def dice_coefficient_all(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection+smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coefficient_all_test(y_true, y_pred, smooth=1.):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = (y_true_f * y_pred_f).sum()
    return (2. * intersection+smooth) / (y_true_f.sum() + y_pred_f.sum() + smooth)

def dice_coefficient_loss(y_true, y_pred):# return average dice loss instead of all
    sum = 0
    class_number = y_true.shape[4]
    for i in range(class_number):
        sum = sum + dice_coefficient_all(y_true[:, :, :, :, i], y_pred[:, :, :, :, i])
    return -sum/class_number
    # return -dice_coefficient(y_true, y_pred)


def weighted_dice_coefficient(y_true, y_pred, axis=(-3, -2, -1), smooth=0.00001):
    """
    Weighted dice coefficient. Default axis assumes a "channels first" data structure
    :param smooth:
    :param y_true:
    :param y_pred:
    :param axis:
    :return:
    """
    return K.mean(2. * (K.sum(y_true * y_pred,
                              axis=axis) + smooth/2)/(K.sum(y_true,
                                                            axis=axis) + K.sum(y_pred,
                                                                               axis=axis) + smooth))


def weighted_dice_coefficient_loss(y_true, y_pred):
    return -weighted_dice_coefficient(y_true, y_pred)


def label_wise_dice_coefficient(y_true, y_pred, label_index):
    return dice_coefficient_all(y_true[:, :, :, :, label_index], y_pred[:, :, :, :, label_index])

def label_wise_dice_coefficient_test(y_true, y_pred, label_index):
    return dice_coefficient_all_test(y_true[:, :, :, :, label_index], y_pred[:, :, :, :, label_index])

def get_label_dice_coefficient_function(label_index):
    f = partial(label_wise_dice_coefficient, label_index=label_index)
    f.__setattr__('__name__', 'label_{0}_dice_coef'.format(label_index))
    return f

dice_coef = dice_coefficient_all
dice_coef_loss = dice_coefficient_loss
