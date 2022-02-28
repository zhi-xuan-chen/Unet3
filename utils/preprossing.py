from __future__ import print_function
from main import n_labels
import numpy as np
import itertools

def convertgt2mask(gt, num_class):
    # gt = np.transpose(gt, [0, 2, 3, 4, 1]) #gt must has only one channel
    shape = list(gt.shape[:-1])
    shape.append(num_class)
    new_mask = np.zeros(shape)
    for i in range(num_class):
        new_mask[gt.squeeze() == i, i] = 1

    # new_mask = np.transpose(new_mask, [0, 4, 1, 2, 3])

    return new_mask

def create_patch_index_list(index_list, image_shape, patch_shape, patch_overlap, patch_start_offset=None):
    patch_index = list()
    for index in index_list:
        if patch_start_offset is not None:
            random_start_offset = np.negative(get_random_nd_index(patch_start_offset))
            patches = compute_patch_indices(image_shape, patch_shape, overlap=patch_overlap, start=random_start_offset)
        else:
            patches = compute_patch_indices(image_shape, patch_shape, overlap=patch_overlap)
        patch_index.extend(itertools.product([index], patches))
    return patch_index

def compute_patch_indices(image_shape, patch_size, overlap, start=None):
    if isinstance(overlap, int):
        overlap = np.asarray([overlap] * len(image_shape))
    if start is None:
        n_patches = np.ceil(image_shape / (patch_size - overlap))
        overflow = (patch_size - overlap) * n_patches - image_shape + overlap
        start = -np.ceil(overflow/2)
    elif isinstance(start, int):
        start = np.asarray([start] * len(image_shape))
    stop = image_shape + start
    step = patch_size - overlap
    return get_set_of_patch_indices(start, stop, step)

def get_set_of_patch_indices(start, stop, step):
    return np.asarray(np.mgrid[start[0]:stop[0]:step[0], start[1]:stop[1]:step[1],
                               start[2]:stop[2]:step[2]].reshape(3, -1).T, dtype=np.int)

def get_random_patch_index(image_shape, patch_shape):
    """
    Returns a random corner index for a patch. If this is used during training, the middle pixels will be seen by
    the model way more often than the edge pixels (which is probably a bad thing).
    :param image_shape: Shape of the image
    :param patch_shape: Shape of the patch
    :return: a tuple containing the corner index which can be used to get a patch from an image
    """
    return get_random_nd_index(np.subtract(image_shape, patch_shape))


def get_random_nd_index(index_max):
    return tuple([np.random.choice(index_max[index] + 1) for index in range(len(index_max))])

def split_patch(data_set, new_shape):
    n = 0
    old_shape = data_set.shape
    new_data_set = np.zeros(new_shape)

    for i in range(old_shape[0]):
        tmp = data_set[i]
        for j in range(old_shape[1]//new_shape[1]):
            for k in range(old_shape[2]//new_shape[2]):
                for m in range(old_shape[3]//new_shape[3]):
                    new_data_set[n] = tmp[new_shape[1] * j: new_shape[1] + new_shape[1] * j,
                                      new_shape[2] * k: new_shape[2] + new_shape[2] * k,
                                      new_shape[3] * m: new_shape[3] + new_shape[3] * m, :]
                    n = n + 1

    return new_data_set

if __name__ == "__main__":
    X_train = np.load('data/MSD Cardiac/train_set.npy')
    Y_train = np.load('data/MSD Cardiac/label_set.npy')
    X_train = X_train[:, 0:128, :, :, :]
    Y_train = Y_train[:, 0:128, :, :, :]
    X_train = split_patch(X_train, (1280, 32, 80, 80, 1))
    Y_train = convertgt2mask(Y_train, n_labels)
    Y_train = split_patch(Y_train, (1280, 32, 80, 80, 2))
    np.save('data/MSD Cardiac/train_set_patch.npy', X_train)
    np.save('data/MSD Cardiac/label_set_patch.npy', Y_train)