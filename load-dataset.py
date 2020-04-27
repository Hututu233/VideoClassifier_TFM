import numpy as np
import os
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
# from keras.applications.mobilenet_v2 import preprocess_input
from keras.applications.inception_v3 import preprocess_input

# BASE_FOLDER = "./datasets/1"
# Tx = 20    # video sequence length

BASE_FOLDER = "./datasets/1_2"
Tx = 40    # video sequence length

# Parameters

nh = 224    # frame height
nw = 224    # frame width
nc = 3      # number of channels

# Dataset preparation
X = []
y = []
i = 1
sample = []
for root, dirs, files in os.walk(BASE_FOLDER):
    for name in files:
        # print(name, root)
        image = load_img(root+os.sep+name, target_size=(nh, nw))

        # # report details about the image
        # print(type(image))
        # print(image.format)
        # print(image.mode)
        # print(image.size)
        # # show the image
        # image.show()

        image = img_to_array(image)
        image = preprocess_input(image)
        sample.append(image)
        if i == Tx:
            X.append(sample)
            if root.endswith("Negative"):
                y.append(0)
            else:
                y.append(1)
            i = 1
            sample = []
        else:
            i = i+1
dataX = np.array(X)
y = np.array(y)
np.save('x.npy', dataX)
np.save('y.npy', y)