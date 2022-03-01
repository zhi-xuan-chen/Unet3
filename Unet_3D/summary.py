import matplotlib.pyplot as plt
import pickle
import numpy as np
from Unet_3D.main import n_labels
from utils.metrics import *
from keras.models import load_model
from preprossing import convertgt2mask
import scipy.misc
import nibabel as nib


def dice_test(X_test_set, Y_test_set, model): #此函数的输入维度，最后一个维度要求是C
    data_shape = Y_test_set.shape
    Y_pred_set = model.predict(X_test_set, batch_size=2, verbose = 1)
    dice_list = [] #存储不同label的dice
    for i in range(data_shape[-1]):
        dice_list[i] = label_wise_dice_coefficient(Y_test_set, Y_pred_set, i)
    dice_average = sum(dice_list) / len(dice_list)
    dice_all = dice_coefficient_all(Y_test_set, Y_pred_set, smooth=0.01)

    return dice_list, dice_average, dice_all

def model_average_dice_test(X_test_set, Y_test_set, num_model, model_path = './model/MSD Cardiac/logs'):
    dice_list = []
    dice_average = []
    dice_all = []

    for k in range(num_model):
        filepath = model_path + '/fold%d_best_model.h5' % k
        model = load_model(filepath)
        dice_list[k], dice_average[k], dice_all[k] = \
            dice_test(X_test_set, Y_test_set, model)

    return np.mean(dice_list, -1), np.mean(dice_average, -1), np.mean(dice_all, -1),

def model_results_display(X_test_set, Y_test_set, patch_shape, num_model, result_path = './Unet_3D/results', model_path = './model/MSD Cardiac/logs'):
    data_shape = X_test_set.shape
    n = 0
    Y_pred_set = [] #用来存放输出的label标签图

    for i in range(data_shape[0]): #从测试数据集中遍历所有patch丢进模型中计算输出
        tmp = X_test_set[i]
        for j in range(data_shape[1]//patch_shape[1]):
            for k in range(data_shape[2]//patch_shape[2]):
                for m in range(data_shape[3]//patch_shape[3]):
                    patch_input = tmp[patch_shape[1] * j: patch_shape[1] + patch_shape[1] * j,
                                      patch_shape[2] * k: patch_shape[2] + patch_shape[2] * k,
                                      patch_shape[3] * m: patch_shape[3] + patch_shape[3] * m, :]
                    patch_output = []
                    for k in range(num_model):
                        filepath = model_path + '/fold%d_best_model.h5' % k
                        model = load_model(filepath)
                        output_tmp = model.predict(patch_input, batch_size=1, verbose=1)
                        patch_output = patch_output + output_tmp
                    patch_output = patch_output / num_model

                    Y_pred_set[i,
                    patch_shape[1] * j: patch_shape[1] + patch_shape[1] * j,
                    patch_shape[2] * k: patch_shape[2] + patch_shape[2] * k,
                    patch_shape[3] * m: patch_shape[3] + patch_shape[3] * m, :] = patch_output.argmax(3)

                    n = n + 1

    Y_pred_set = convertgt2mask(Y_pred_set, n_labels)
    Y_test_set = convertgt2mask(Y_test_set, n_labels)

    label_shape = Y_test_set.shape
    num_channel = data_shape[-1]
    num_label = label_shape[-1]
    for i in range(num_label):
        gt_mask = np.expand_dims(Y_test_set[:,:,:,:,i],-1).repeat(num_channel,axis=-1)
        pred_mask = np.expand_dims(Y_pred_set[:,:,:,:,i],-1).repeat(num_channel,axis=-1)
        gt_result = X_test_set * gt_mask
        pred_results = X_test_set * pred_mask

        for n in range(data_shape[0]):  # 这里需要修改
            gt_image = nib.Nifti1Image(gt_result[n], np.eye(4)) #S*H*W*C
            gt_image.set_data_dtype(np.my_dtype)
            nib.save(gt_image, result_path + './gt/lable_%d_gt_result_%d.nii.gz' % i % n)

            pred_image = nib.Nifti1Image(pred_results[n], np.eye(4)) #S*H*W*C
            pred_image.set_data_dtype(np.my_dtype)
            nib.save(pred_image, result_path + './pred/lable_%d_pred_result_%d.nii.gz' % i % n)
    return


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
    dice_list, dice_average, dice_all = model_average_dice_test(X_test_set_patch, Y_test_set_patch, 5)
    print('dice_list is:' + str(dice_list))
    print('dice_average is:' + str(dice_average))
    print('dice_all is:' + str(dice_all))

    X_test_set = np.load('./data/MSD Cardiac/train_set.npy')
    Y_test_set = np.load('./data/MSD Cardiac/label_set.npy')
    X_test_set = X_test_set[:, 0:128, :, :, :]# crop
    Y_test_set = Y_test_set[:, 0:128, :, :, :]

    model_results_display(X_test_set, Y_test_set, (1280, 32, 80, 80, 1), num_model = 5)




