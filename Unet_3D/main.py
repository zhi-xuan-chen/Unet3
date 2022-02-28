from Unet_3D.model import unet3
from sklearn.model_selection import KFold
import keras.backend as K
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
import os
import pickle

def scheduler(epoch):
    # 每隔100个epoch，学习率减小为原来的1/10
    if epoch % 100 == 0 and epoch != 0:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr / 10)
        print("lr changed to {}".format(lr / 10))
    else:
        lr = K.get_value(model.optimizer.lr)
        print("lr is{}".format(lr / 10))
    return K.get_value(model.optimizer.lr)


if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    X_train = np.load('./data/MSD Cardiac/train_set_patch.npy')  # patched dataset
    Y_train = np.load('./data/MSD Cardiac/label_set_patch.npy')
    kf = KFold(n_splits=5, shuffle=True, random_state=1000)
    n_labels = 2  # Class number
    lr = 0.001
    depth = 5
    n_base_filters = 4  # Channel number at level0
    k = 0

    for idxtr, idxts in kf.split(X_train, Y_train):
        model = unet3.unet_model_3d(input_shape=(32, 80, 80, 1), n_labels=n_labels, initial_learning_rate=lr,
                                    depth=depth, n_base_filters=n_base_filters, include_label_wise_dice_coefficients=True)
        model.summary()
        print("Now training fold%d" % k)
        reduce_lr = LearningRateScheduler(scheduler)
        checkpoint = ModelCheckpoint(filepath='./model/MSD Cardiac/logs/fold%d_best_model.h5' % k, monitor='val_loss', mode='auto',
                                     save_best_only='True')
        history = model.fit(X_train[idxtr], Y_train[idxtr],  validation_data= (X_train[idxts],Y_train[idxts]),
                                    batch_size=16, epochs=200, callbacks=[reduce_lr, checkpoint])
        #model.save('./model/MSD Cardiac/fold%d.h5' % k)
        with open("./model/MSD Cardiac/logs/fold%d_log.txt" % k, "wb") as file:
            pickle.dump(history.history, file)
        k = k + 1



