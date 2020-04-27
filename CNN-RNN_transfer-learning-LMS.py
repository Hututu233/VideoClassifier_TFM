import math
import fnmatch
from keras.models import Model
from keras.layers import Dense, Input, Lambda, Dropout, CuDNNGRU, CuDNNLSTM
from keras.layers import Flatten
from keras.layers.recurrent import LSTM, GRU
from keras.optimizers import SGD
from keras.utils import plot_model
from keras.layers import TimeDistributed
from sklearn.datasets import make_classification
from keras.applications import MobileNetV2
from keras.applications import inception_v3
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from numpy import savetxt
import keras.backend.tensorflow_backend as K
import tensorflow as tf

def generator(dataset_folder, samples_per_epoch, batch_size):
  number_of_batches = samples_per_epoch/batch_size
  batches_counter = 0
  sample_counter = 0
  while 1:
    X_batch = []
    y_batch = []
    if batches_counter < math.floor(number_of_batches):
        for i in range(batch_size):
            X_batch.append(np.load('T:/dekra/video-classifer/cache/' + dataset_folder + '/X' + str(sample_counter) + '.npy'))
            y_batch.append(np.load('T:/dekra/video-classifer/cache/' + dataset_folder + '/y' + str(sample_counter) + '.npy'))
            sample_counter += 1
    else:   # last batch
        for i in range(samples_per_epoch - batches_counter*batch_size):
            X_batch.append(np.load('T:/dekra/video-classifer/cache/' + dataset_folder + '/X' + str(sample_counter) + '.npy'))
            y_batch.append(np.load('T:/dekra/video-classifer/cache/' + dataset_folder + '/y' + str(sample_counter) + '.npy'))
            sample_counter += 1
    batches_counter += 1
    yield np.array(X_batch), np.array(y_batch)
    #restart counter to yeild data in the next epoch as well
    if batches_counter >= number_of_batches: # math.floor(number_of_batches)
        batches_counter = 0
        sample_counter = 0


## Parameters
Tx = 140     # video sequence length
nh = 224    # frame height
nw = 224    # frame width
nc = 3      # number of channels
train_size = len(fnmatch.filter(os.listdir('T:/dekra/video-classifer/cache/train'), 'X*.npy'))
val_size = len(fnmatch.filter(os.listdir('T:/dekra/video-classifer/cache/val'), 'X*.npy'))
batch_size = 8

# CNN-based Feature Extractor (no trainable part)
# cnn_model = MobileNetV2(include_top=False, input_shape=(nh, nw, nc))
cnn_model = inception_v3.InceptionV3(include_top=False, input_shape=(nh, nw, nc))
for layer in cnn_model.layers:
    layer.trainable = False
cnn_out = Flatten()(cnn_model.outputs)
cnn = Model(inputs=cnn_model.input, outputs=cnn_out)

# Video Classifier (trainable part)
video_input = Input(shape=(Tx, nh, nw, nc))
video_sequence = TimeDistributed(cnn)(video_input)  # the output will be a sequence of vectors
hidden1 = CuDNNLSTM(32, return_sequences=False)(video_sequence)
drop = Dropout(0.2)(hidden1)
hidden2 = Dense(64, activation='relu')(drop)
output = Dense(1, activation='sigmoid')(hidden2)
model = Model([video_input], outputs=output)
opt = SGD(lr=0.001, momentum=0.9)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
# summarize layers
print(cnn.summary())
print(model.summary())
print("Steps per epoch: " + str(math.ceil(train_size/batch_size)))
#TRAINING
history = model.fit_generator(generator('train', train_size, batch_size),
                              validation_data=(generator('val',val_size, batch_size)),
                              epochs=10, steps_per_epoch=math.ceil(train_size/batch_size),
                              validation_steps=math.ceil(val_size/batch_size))

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

#EVALUATION
# Load evaluation data --> in future use a "Test set"
evaly = []
for i in range(val_size):
    evaly.append(np.load('T:/dekra/video-classifer/cache/val/y' + str(i) + '.npy'))

# nn_probs = model.predict(np.array(evalX), batch_size=batch_size)
nn_probs = model.predict_generator(generator('val', val_size, batch_size), steps=math.ceil(val_size/batch_size))
# savetxt('data.csv', nn_probs, delimiter=',')
nn_yhat = (nn_probs > 0.5).astype(np.int)
nn_precision, nn_recall, _ = precision_recall_curve(np.array(evaly), nn_probs)
nn_f1, nn_auc = f1_score(np.array(evaly), nn_yhat), auc(nn_recall, nn_precision)
# summarize scores
print('NN: f1=%.3f auc=%.3f' % (nn_f1, nn_auc))
# plot the precision-recall curves
plt.plot(nn_recall, nn_precision, marker='.', label='NN')
# axis labels
plt.xlabel('Recall')
plt.ylabel('Precision')
# show the legend
plt.legend()
# show the plot
plt.show()

tn, fp, fn, tp = confusion_matrix(np.array(evaly), nn_yhat).ravel()
print("True Positives: %d " % tp)
print("True Negatives: %d " % tn)
print("False Positives: %d " % fp)
print("False Negatives: %d " % fn)

# SAVE MODEL INTO FILE
model.save("model.h5")
