from numpy import array
from numpy import hstack
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense, Conv2D, MaxPooling2D, LSTM, Flatten, TimeDistributed, GRU
from keras.models import Sequential
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.merge import concatenate
from sklearn.datasets import make_classification
import keras.backend as K
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Parameters
Tx = 40     # video sequence length
nh = 224    # frame height
nw = 224    # frame width
nc = 3      # number of channels


def generator(X_data, y_data, batch_size):

  samples_per_epoch = X_data.shape[0]
  number_of_batches = samples_per_epoch/batch_size
  counter=0

  while 1:

    X_batch = np.array(X_data[batch_size*counter:batch_size*(counter+1)]).astype('float32')
    y_batch = np.array(y_data[batch_size*counter:batch_size*(counter+1)]).astype('float32')
    counter += 1
    yield X_batch,y_batch

    #restart counter to yeild data in the next epoch as well
    if counter >= number_of_batches:
        counter = 0


dataX = np.load('x.npy')
dataX = dataX/255.0
y = np.load('y.npy')

trainX, valX, trainy, valy = train_test_split(dataX, y, test_size=0.3, random_state=2, shuffle=True, stratify=y)
unique_elements, counts_elements = np.unique(trainy, return_counts=True)
print("Training:")
print(np.asarray((unique_elements, counts_elements)))
unique_elements, counts_elements = np.unique(valy, return_counts=True)
print("Val:")
print(np.asarray((unique_elements, counts_elements)))

model = Sequential()
model.add(TimeDistributed(Conv2D(64, 5, activation='relu', padding='same', name='conv1', data_format='channels_last', input_shape=(nh, nw, nc))))
model.add(TimeDistributed(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same', name='pool1')))

model.add(TimeDistributed(Conv2D(64, 5, activation='relu', padding='same', name='conv2')))
model.add(TimeDistributed(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same', name='pool2')))
#
model.add(TimeDistributed(Conv2D(64, 5, activation='relu', padding='same', name='conv3')))
model.add(TimeDistributed(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same', name='pool3')))

model.add(TimeDistributed(Conv2D(64, 5, activation='relu', padding='same', name='conv4')))
model.add(TimeDistributed(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same', name='pool4')))

model.add(TimeDistributed(Flatten()))
model.add(LSTM(32, return_sequences=False, dropout=0.5))
model.add(Dense(1, activation='sigmoid'))

# summarize layers
model.build(input_shape=(None, Tx, nh, nw, nc))
print(model.summary())

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#TRAINING
# history = model.fit(trainX, trainy, epochs=10, validation_data=(valX, valy), batch_size=4)
batch_size = 6
history = model.fit_generator(generator(trainX, trainy, batch_size),
                              validation_data=(generator(valX, valy, batch_size)),
                              epochs=2, steps_per_epoch=trainX.shape[0]//batch_size,
                              validation_steps=valX.shape[0]//batch_size)
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


evalX = valX
evaly = valy
#EVALUATION
nn_probs = model.predict(evalX, batch_size=4)
# savetxt('data.csv', nn_probs, delimiter=',')
nn_yhat = (nn_probs > 0.5).astype(np.int)
nn_precision, nn_recall, _ = precision_recall_curve(evaly, nn_probs)
nn_f1, nn_auc = f1_score(evaly, nn_yhat), auc(nn_recall, nn_precision)
# summarize scores
print('NN: f1=%.3f auc=%.3f' % (nn_f1, nn_auc))
# plot the precision-recall curves
no_skill = len(y[y==1]) / len(y)
plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
plt.plot(nn_recall, nn_precision, marker='.', label='NN')
# axis labels
plt.xlabel('Recall')
plt.ylabel('Precision')
# show the legend
plt.legend()
# show the plot
plt.show()

tn, fp, fn, tp = confusion_matrix(evaly, nn_yhat).ravel()
print("True Positives: %d " % tp)
print("True Negatives: %d " % tn)
print("False Positives: %d " % fp)
print("False Negatives: %d " % fn)