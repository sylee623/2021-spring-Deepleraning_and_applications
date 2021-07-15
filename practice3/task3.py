# -*- coding: utf-8 -*-
"""Untitled20.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1zmd3B3DymFlkLBpv2RwoEfKrnCQpVh3h
"""

import matplotlib.pyplot as plt
import numpy as np
import keras
import time
import tensorflow as tf

path = "/content/"

x1_train = open(path + "/train_x1.txt", 'r')
x1_train = x1_train.readlines()
x1_train = list(map(float,x1_train))
x2_train = open(path + "/train_x2.txt", 'r')
x2_train = x2_train.readlines()
x2_train = list(map(float,x2_train))
y_train = open(path + "/train_y.txt", 'r')
y_train = y_train.readlines()
y_train = list(map(float,y_train))

x1_test = open(path + "/test_x1.txt", 'r')
x1_test = x1_test.readlines()
x1_test = list(map(float,x1_test))
x2_test = open(path + "/test_x2.txt", 'r')
x2_test = x2_test.readlines()
x2_test = list(map(float,x2_test))
y_test = open(path + "/test_y.txt", 'r')
y_test = y_test.readlines()
y_test = list(map(float,y_test))

print(len(x1_train))

x_train = np.array([x1_train, x2_train]).T
y_train = np.array(y_train)
x_test = np.array([x1_test, x2_test]).T
y_test = np.array(y_test)

lr = 1e-3
K = 1000
m = 10000
n = 500


model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(2, 1)),
      tf.keras.layers.Dense(3, activation='sigmoid'),
      tf.keras.layers.Dense(1, activation='sigmoid')
])

opt = keras.optimizers.SGD(learning_rate = lr)
model.compile(optimizer=opt,
              loss='binary_crossentropy',
              metrics=['accuracy'])

class LossHistory(tf.keras.callbacks.Callback): #참고 한 사이트 : https://stackoverflow.com/questions/56354580/log-keras-metrics-for-each-batch-as-per-keras-example-for-the-loss
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

batch32_SGD_CE_loss_history = LossHistory()

x_train = np.array([x1_train, x2_train]).T
y_train = np.array(y_train)

start = time.time()

batch32_SGD_CE_history = model.fit(x_train, y_train, epochs= K , callbacks=[batch32_SGD_CE_loss_history], batch_size = 32)

end = time.time()

print('Training time :' , end - start)

batch32_SGD_CE.evaluate(x_test, y_test)

batch_loss = plt.title('batch 32, cross eentropy, SGD, batch loss')
batch_loss = plt.plot(batch32_SGD_CE_loss_history.losses)

epoch_loss = plt.title('batch 32, cross eentropy, SGD, loss')
epoch = list(range(1,1001))
epoch_loss = plt.plot(epoch, batch32_SGD_CE_history.history['loss'])

model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(2, 1)),
      tf.keras.layers.Dense(3, activation='sigmoid'),
      tf.keras.layers.Dense(1, activation='sigmoid')
])

opt = keras.optimizers.SGD(learning_rate = lr)
model.compile(optimizer=opt,
              loss='MeanSquaredError',
              metrics=['accuracy'])

# x_train = np.array([x1_train, x2_train]).T
# y_train = np.array(y_train)
batch32_SGD_MSE_loss_history = LossHistory()

start = time.time()
batch32_SGD_MSE_history = model.fit(x_train, y_train, epochs= K , callbacks=[batch32_SGD_MSE_loss_history], batch_size = 32)
end = time.time()

print('Training time : ' , end - start)

batch_loss = plt.title('batch 32, MSE, SGD, batch loss')
batch_loss = plt.plot(batch32_SGD_MSE_loss_history.losses)

epoch_loss = plt.title('MSE vs CE loss')
epoch = list(range(1,1001))
epoch_loss = plt.plot(epoch, batch32_SGD_MSE_history.history['loss'], epoch, batch32_SGD_CE_history.history['loss'])

epoch_accuracy = plt.title('MSE vs CE accuracy')
epoch = list(range(1,1001))
epoch_accuracy = plt.plot(epoch, batch32_SGD_MSE_history.history['accuracy'], epoch, batch32_SGD_CE_history.history['accuracy'])

x_test = np.array([x1_test, x2_test]).T
y_test = np.array(y_test)
model.evaluate(x_test, y_test)



model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(2, 1)),
      tf.keras.layers.Dense(3, activation='sigmoid'),
      tf.keras.layers.Dense(1, activation='sigmoid')
])

opt = keras.optimizers.RMSprop(learning_rate = lr)
model.compile(optimizer=opt,
              loss='binary_crossentropy',
              metrics=['accuracy'])

batch32_RMSprop_CE_loss_history = LossHistory()

start = time.time()
x_train = np.array([x1_train, x2_train]).T
y_train = np.array(y_train)
batch32_RMSprop_CE_history = model.fit(x_train, y_train, epochs= K , batch_size=32, callbacks=[batch32_RMSprop_CE_loss_history])

end = time.time()

print('Training time :' , end - start)

model.evaluate(x_test, y_test)

batch_loss = plt.title('batch 32, CE, RMSProp, batch loss')
batch_loss = plt.plot(batch32_RMSprop_CE_loss_history.losses)



model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(2, 1)),
      tf.keras.layers.Dense(3, activation='sigmoid'),
      tf.keras.layers.Dense(1, activation='sigmoid')
])

batch32_Adam_CE_loss_history = LossHistory()

opt = keras.optimizers.Adam(learning_rate = lr)
model.compile(optimizer=opt,
              loss='binary_crossentropy',
              metrics=['accuracy'])

start = time.time()

batch32_Adam_CE_history = model.fit(x_train, y_train, epochs= K , batch_size = 32, callbacks = [batch32_Adam_CE_loss_history])

end = time.time()

print('Training time :' , end - start)

model.evaluate(x_test, y_test)

batch_loss = plt.title('batch 32, CE, Adam, batch loss')
batch_loss = plt.plot(batch32_Adam_CE_loss_history.losses)

epoch_loss = plt.title('SGD vs RMSProp vs Adam loss')
epoch = list(range(1,1001))
epoch_loss = plt.plot(epoch, batch32_Adam_CE_history.history['loss'] ,label = 'Adam')
epoch_loss = plt.plot(epoch, batch32_SGD_CE_history.history['loss'] ,label = 'SGD')
epoch_loss = plt.plot(epoch, batch32_RMSprop_CE_history.history['loss'] ,label = 'RMSprop')
epoch_loss = plt.legend()

epoch_loss = plt.title('SGD vs RMSProp vs Adam accuracy')
epoch = list(range(1,1001))
epoch_loss = plt.plot(epoch, batch32_Adam_CE_history.history['accuracy'] ,label = 'Adam')
epoch_loss = plt.plot(epoch, batch32_SGD_CE_history.history['accuracy'] ,label = 'SGD')
epoch_loss = plt.plot(epoch, batch32_RMSprop_CE_history.history['accuracy'] ,label = 'RMSprop')
epoch_loss = plt.legend()



model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(2, 1)),
      tf.keras.layers.Dense(3, activation='sigmoid'),
      tf.keras.layers.Dense(1, activation='sigmoid')
])

opt = keras.optimizers.SGD(learning_rate = lr)

batch4_SGD_CE_loss_history = LossHistory()

model.compile(optimizer=opt,
              loss='binary_crossentropy',
              metrics=['accuracy'])

start = time.time()
x_train = np.array([x1_train, x2_train]).T
y_train = np.array(y_train)
batch4_SGD_CE_history = model.fit(x_train, y_train, epochs= K , batch_size = 4, callbacks=[batch4_SGD_CE_loss_history])

end = time.time()

print('Training time :' , end - start)

batch_loss = plt.title('batch 4, CE, SGD, batch loss')
batch_loss = plt.plot(batch4_SGD_CE_loss_history.losses)

model.evaluate(x_test, y_test)




model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(2, 1)),
      tf.keras.layers.Dense(3, activation='sigmoid'),
      tf.keras.layers.Dense(1, activation='sigmoid')
])


batch128_SGD_CE_loss_history = LossHistory()

opt = keras.optimizers.SGD(learning_rate = lr)
model.compile(optimizer=opt,
              loss='binary_crossentropy',
              metrics=['accuracy'])

start = time.time()
x_train = np.array([x1_train, x2_train]).T
y_train = np.array(y_train)
batch128_SGD_CE_history = model.fit(x_train, y_train, epochs= K , batch_size = 128, callbacks = [batch128_SGD_CE_loss_history ])

end = time.time()

print('Training time :' , end - start)

model.evaluate(x_test, y_test)

batch_loss = plt.title('batch 128, CE, SGD, batch loss')
batch_loss = plt.plot(batch128_SGD_CE_loss_history.losses)

epoch_loss = plt.title('4 vs 32 vs 128 accuracy')
epoch = list(range(1,1001))
epoch_loss = plt.plot(epoch, batch4_SGD_CE_history.history['accuracy'] ,label = '4')
epoch_loss = plt.plot(epoch, batch32_SGD_CE_history.history['accuracy'] ,label = '32')
epoch_loss = plt.plot(epoch, batch128_SGD_CE_history.history['accuracy'] ,label = '128')
epoch_loss = plt.legend()

epoch_loss = plt.title('4 vs 32 vs 128 loss')
epoch = list(range(1,1001))
epoch_loss = plt.plot(epoch, batch4_SGD_CE_history.history['loss'] ,label = '4')
epoch_loss = plt.plot(epoch, batch32_SGD_CE_history.history['loss'] ,label = '32')
epoch_loss = plt.plot(epoch, batch128_SGD_CE_history.history['loss'] ,label = '128')
epoch_loss = plt.legend()
