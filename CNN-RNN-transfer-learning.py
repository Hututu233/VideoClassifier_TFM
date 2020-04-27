
from keras.models import Model
from keras.layers import Dense, Input, Lambda, Dropout
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

# Parameters
Tx = 60     # video sequence length
nh = 224    # frame height
nw = 224    # frame width
nc = 3      # number of channels

dataX = np.load('x.npy')
y = np.load('y.npy')

trainX, valX, trainy, valy = train_test_split(dataX, y, test_size=0.3, random_state=2, shuffle=True, stratify=y)
unique_elements, counts_elements = np.unique(trainy, return_counts=True)
print("Training:")
print(np.asarray((unique_elements, counts_elements)))
unique_elements, counts_elements = np.unique(valy, return_counts=True)
print("Val:")
print(np.asarray((unique_elements, counts_elements)))



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
hidden1 = GRU(32, return_sequences=False)(video_sequence)
drop = Dropout(0.2)(hidden1)
hidden2 = Dense(64, activation='relu')(drop)
output = Dense(1, activation='sigmoid')(hidden2)
model = Model([video_input], outputs=output)
opt = SGD(lr=0.001, momentum=0.9)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
# summarize layers
print(cnn.summary())
print(model.summary())

# X, y = make_classification(n_samples=100, n_classes=2, random_state=1, n_features=Tx*nh*nw*nc)
# trainX = X.reshape((X.shape[0], Tx, nh, nw, nc))

#TRAINING
history = model.fit(trainX, trainy, epochs=10, validation_data=(valX, valy), shuffle=True, batch_size=12)
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
nn_probs = model.predict(evalX, batch_size=12)
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