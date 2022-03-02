import matplotlib.pyplot as plt
import pickle
import numpy as np
from Unet_3D.main import n_labels
from utils.metrics import *
from keras.models import load_model
from preprossing import convertgt2mask
import scipy.misc
import nibabel as nib

# X_test_set and Y_test_set are all patch size

def dice_test(X_test_set, Y_test_set, origin_shape, model): #此函数的输入维度，最后一个维度要求是C
    n, s, h, w, c = X_test_set.shape
    N, S, H, W, C = origin_shape
    num_patch = (S * H * W) / \
                (s * h * w)  # number of patch in one origin sample
    # X_test_set 应该包含所有的数据，但是可以慢慢丢入模型，防止溢出
    Y_pred_set = np.zeros(Y_test_set.shape)
    for i in range(N):
        Y_pred_set[i*num_patch : i*num_patch+num_patch] = \
            model.predict(X_test_set[i*num_patch : i*num_patch+num_patch], batch_size=2, verbose = 1)


def model_results_display(X_test_set, origin_shape, model): #X_test_set is all patch
    n, s, h, w, c = X_test_set.shape
    N, S, H, W, C = origin_shape
    num_patch = (S * H * W) / \
                (s * h * w) # number of patch in one origin sample

    Y_pred_set = np.zeros(origin_shape)

    for i in range(origin_shape[0]):
        output_tmp = model.predict(X_test_set[i*num_patch : i*num_patch+num_patch], batch_size=2, verbose = 1)

        for j in range(S // s):
            for k in range(H // h):
                for m in range(W // w):
                    Y_pred_set[i, s * j: s + s * j,
                                      h * k: h + h * k,
                                      w * m: w + w * m, :] = output_tmp[j * (H//h) * (W//w) + k * (W//w) + m]

    Y_pred_label = Y_pred_set.argmax(4) #存放输出原始大小的label标签图

    return Y_pred_label


def model_test(X_test_set, Y_test_set, origin_shape, num_model, model_path = './model/MSD Cardiac/logs'):
    dice_list = []
    dice_average = []
    # dice_all = []

    k_max = 0
    dice_max = 0

    for k in range(num_model):
        filepath = model_path + '/fold%d_best_model.h5' % k
        model = load_model(filepath, compile=False)
        dice_list_tmp, dice_average_tmp = dice_test(X_test_set, origin_shape, model)
        if dice_average_tmp > dice_max:
            k_max = k
            dice_max = dice_average_tmp
        dice_list.append(dice_list_tmp)
        dice_average.append(dice_average_tmp)

    best_filepath = model_path + '/fold%d_best_model.h5' % k_max
    model = load_model(best_filepath, compile=False)
    dice_list, dice_average = dice_test(X_test_set, Y_test_set, origin_shape, model)
    Y_pred_label = model_results_display(X_test_set, Y_test_set, )

    return np.mean(dice_list, -1), np.mean(dice_average, -1), Y_pred_label#, np.mean(dice_all, -1),

if __name__ == "__main__":
    kfold = 5
    for k in range(kfold):
        with open("./model/MSD Cardiac/logs/fold%d_log.txt" % k, "rb") as file:
            history = pickle.load(file)
        plt.plot(history['loss'])
        plt.plot(history['val_loss'])
        plt.title("fold%d model loss pic" % k)
        plt.ylabel("avg dice loss")
        plt.xlabel("epoch")
        plt.legend(["train", "test"],loc="lower right")
        plt.savefig("./model/MSD Cardiac/logs/fold%d loss pic.jpg" % k)
        plt.show()

    X_test_set_patch = np.load('./data/MSD Cardiac/train_set_patch.npy')
    Y_test_set_patch = np.load('./data/MSD Cardiac/label_set_patch.npy')

    dice_list, dice_average, Y_pred_label = \
        model_test(X_test_set_patch, Y_test_set_patch, (20,128,320,320,1), 5, model_path='./model/MSD Cardiac/logs')

    print('dice_list is:' + str(dice_list))
    print('dice_average is:' + str(dice_average))






