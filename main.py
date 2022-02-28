from model import unet3
from sklearn.model_selection import KFold
import keras.backend as K
from tensorflow.keras.callbacks import LearningRateScheduler
import numpy as np
import os
import keras
from keras.models import load_model

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
X_train = np.load('data/MSD Cardiac/train_set_patch.npy')
Y_train = np.load('data/MSD Cardiac/label_set_patch.npy')
n_labels = 2 # Class number
lr = 0.0001
depth = 5
n_base_filters = 4 # Channel number at level0
kf = KFold(n_splits=5, shuffle = True, random_state=1000)
k = 0

for idxtr, idxts in kf.split(X_train, Y_train):
    model = unet3.unet_model_3d(input_shape=(32, 80, 80, 1), n_labels=n_labels, initial_learning_rate=lr,
                                depth = depth, n_base_filters = n_base_filters)
    model.summary()
    print("Now training fold%d" % k)
    model.fit(X_train[idxtr], Y_train[idxtr], batch_size=16, epochs=200)
    #y_pred = model.predict(X_train[:1])
    model.save('./model/MSD Cardiac/fold%d.h5' % k)
    k = k + 1



