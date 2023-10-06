# RNN on IMDb

# Import libraries
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Embedding, SimpleRNN
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.utils import to_categorical

# Loading dataset
max_features = 10000
max_len = 100
(x_train,y_train), (x_test,y_test) = imdb.load_data(num_words=max_features)
x_train = sequence.pad_sequences(x_train, maxlen=max_len)
x_test = sequence.pad_sequences(x_test, maxlen=max_len)
y_train = to_categorical(y_train,2)
y_test = to_categorical(y_test,2)

# Building model

model = Sequential()
model.add(Embedding(max_features,32,max_len))
model.add(SimpleRNN(32, activation='relu'))
model.add(Dense(2,activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])

# Train Model

history = model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.2)

# Plotting results

import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
epochs = range(1,len(acc)+1)

plt.plot(epochs,acc,'bo',label='Training Accuracy')

plt.plot(epochs,val_acc,'b',label='Validation Accuracy')

plt.title('Training and Validation Accuracy')

plt.legend()

plt.figure()

plt.show()