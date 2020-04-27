import matplotlib.pyplot as plt
import numpy as np
import os
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
# from keras.applications.mobilenet_v2 import preprocess_input
from keras.applications.inception_v3 import preprocess_input

BASE_FOLDER = "T:/dekra/video-classifer/expreriment2"
Tx = 60    # video sequence length
step_size = 30

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
for root, dirs, files in os.walk(BASE_FOLDER):
    for name in files:
        # print(name, root)
        image = load_img(root+os.sep+name, target_size=(nh, nw))
        image = img_to_array(image)
        image = preprocess_input(image)
        if root.endswith("positive"):
            pos_sample.append(image)
        else:
            neg_sample.append(image)

X_pos = np.array(pos_sample)
X_pos_win = as_sliding_window(X_pos, Tx, step_size=step_size)
Y_pos = np.ones(X_pos_win.shape[0])
X_neg = np.array(neg_sample)
X_neg_win = as_sliding_window(X_neg, Tx, step_size=step_size)
Y_neg = np.zeros(X_neg_win.shape[0])

X = np.concatenate((X_pos_win, X_neg_win))
y = np.concatenate((Y_pos, Y_neg))

np.save('x.npy', X)
np.save('y.npy', y)

print(X.shape)
print(y.shape)