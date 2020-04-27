from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import os
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
# from keras.applications.mobilenet_v2 import preprocess_input
from keras.applications.inception_v3 import preprocess_input

BASE_FOLDER = "T:/dekra/video-classifer/experiment19"

Tx = 140    # video sequence length
step_size = 140
test_mode = False

if test_mode is True:
    step_size = Tx

# Parameters
nh = 224    # frame height
nw = 224    # frame width
nc = 3      # number of channels

def as_sliding_window(x, window_size, axis=0, window_axis=None,
                      subok=False, writeable=True, step_size=1):
    """
    Make a sliding window across an axis.

    Uses ``numpy.lib.stride_tricks.as_strided``, similar caveats apply.

    Parameters
    ----------
    x : array_like
        Array from where the sliding window is created.
    window_size: int
        Size of the sliding window.
    axis: int
        Dimension across which the sliding window is created.
    window_axis: int
        New dimension for the sliding window. By default, the new
        dimension is inserted before ``axis``.
    subok: bool
        If True, subclasses are preserved
        (see ``numpy.lib.stride_tricks.as_strided``).
    writeable: bool
        If set to False, the returned array will always be readonly.
        Otherwise it will be writable if the original array was. It
        is advisable to set this to False if possible
        (see ``numpy.lib.stride_tricks.as_strided``).

    Returns
    --------
    sliding_window: ndarray
        View of the given array as a sliding window along ``axis``.
    """
    from numpy.lib.stride_tricks import as_strided
    x = np.asarray(x)
    axis %= x.ndim
    if window_axis is None:
        window_axis = axis
    window_axis %= x.ndim + 1
    # Make shape
    shape = list(x.shape)
    n = shape[axis]
    shape[axis] = window_size
    shape.insert(window_axis, max(n - window_size + 1, 0))
    # Make strides
    strides = list(x.strides)
    strides.insert(window_axis, strides[axis])
    # Make sliding window view
    sliding_window = as_strided(x, shape, strides,
                                subok=subok, writeable=writeable)[0::step_size]
    return sliding_window

neg_sample = []
pos_sample = []
neg_test_sample = []
pos_test_sample = []

for root, dirs, files in os.walk(BASE_FOLDER):
    for name in files:
        if "train_val" in root:
            if root.endswith("positive"):
                pos_sample.append(root+os.sep+name)
            else:
                neg_sample.append(root+os.sep+name)
        if "test" in root:
            if root.endswith("positive"):
                pos_test_sample.append(root + os.sep + name)
            else:
                neg_test_sample.append(root + os.sep + name)

# Train and Validation sets
if not test_mode:
    X_pos = np.array(pos_sample)
    X_pos_win = as_sliding_window(X_pos, Tx, step_size=step_size)
    Y_pos = np.ones(X_pos_win.shape[0])
    X_neg = np.array(neg_sample)
    X_neg_win = as_sliding_window(X_neg, Tx, step_size=step_size)
    Y_neg = np.zeros(X_neg_win.shape[0])
    X = np.concatenate((X_pos_win, X_neg_win))
    y = np.concatenate((Y_pos, Y_neg))
    trainX, valX, trainy, valy = train_test_split(X, y, test_size=0.3, random_state=2, shuffle=True, stratify=y)
    unique_elements, counts_elements = np.unique(trainy, return_counts=True)
    print("Training:")
    print(np.asarray((unique_elements, counts_elements)))
    unique_elements, counts_elements = np.unique(valy, return_counts=True)
    print("Val:")
    print(np.asarray((unique_elements, counts_elements)))
    for i in range(trainX.shape[0]):
        image_list = []
        for j in range(trainX.shape[1]):
            image = load_img(trainX[i, j], target_size=(nh, nw))
            image = img_to_array(image)
            image = preprocess_input(image)
            image_list.append(image)
        np.save('T:/dekra/video-classifer/cache/train/X' + str(i) + '.npy', np.array(image_list))
        np.save('T:/dekra/video-classifer/cache/train/y' + str(i) + '.npy', trainy[i])
    for i in range(valX.shape[0]):
        image_list = []
        for j in range(valX.shape[1]):
            image = load_img(valX[i, j], target_size=(nh, nw))
            image = img_to_array(image)
            image = preprocess_input(image)
            image_list.append(image)
        np.save('T:/dekra/video-classifer/cache/val/X' + str(i) + '.npy', np.array(image_list))
        np.save('T:/dekra/video-classifer/cache/val/y' + str(i) + '.npy', valy[i])

# Test set
if test_mode:
    X_pos = np.array(pos_test_sample)
    X_pos_win = as_sliding_window(X_pos, Tx, step_size=step_size)
    Y_pos = np.ones(X_pos_win.shape[0])
    X_neg = np.array(neg_test_sample)
    X_neg_win = as_sliding_window(X_neg, Tx, step_size=step_size)
    Y_neg = np.zeros(X_neg_win.shape[0])
    testX = np.concatenate((X_pos_win, X_neg_win))
    testy = np.concatenate((Y_pos, Y_neg))
    unique_elements, counts_elements = np.unique(testy, return_counts=True)
    print("Test:")
    print(np.asarray((unique_elements, counts_elements)))
    for i in range(testX.shape[0]):
        image_list = []
        for j in range(testX.shape[1]):
            image = load_img(testX[i, j], target_size=(nh, nw))
            image = img_to_array(image)
            image = preprocess_input(image)
            image_list.append(image)
        np.save('T:/dekra/video-classifer/cache/test/X' + str(i) + '.npy', np.array(image_list))
        np.save('T:/dekra/video-classifer/cache/test/y' + str(i) + '.npy', testy[i])

