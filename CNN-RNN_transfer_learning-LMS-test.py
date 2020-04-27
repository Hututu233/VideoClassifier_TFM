import fnmatch
import os
import numpy as np
from keras.models import load_model
import time

# SAVE MODEL INTO FILE
model = load_model("model.h5")

test_size = len(fnmatch.filter(os.listdir('data/test'), 'X*.npy'))

#EVALUATION
testX = []
testy = []
for i in range(test_size):
    testX.append(np.load('data/test/X' + str(i) + '.npy'))
    testy.append(np.load('data/test/y' + str(i) + '.npy'))

X = np.array(testX)
y = np.array(testy)

for i in range(test_size):
    start_time = time.time()
    nn_probs = model.predict(np.expand_dims(testX[i], axis=0))
    nn_yhat = (nn_probs > 0.5).astype(np.int)
    print("True: " + str(y[i]) + " Predicted: " + str(nn_yhat) + " in %.3f seconds" % (time.time() - start_time))
