import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy 
import cv2

mnist = keras.datasets.mnist
(x_train,y_train),(x_test,y_test) = mnist.load_data()
print(x_train[0].shape)
x_train,x_test = x_train/255.0,x_test/255.0

model = keras.Sequential()
# model.add(keras.Input(shape = (28,28)))
model.add(keras.layers.Flatten(input_shape = (28,28)))
model.add(keras.layers.Dense(128,activation = 'relu'))
model.add(keras.layers.Dense(128,activation = 'relu'))
model.add(keras.layers.Dense(10,activation = 'softmax' ))
model.summary()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train,y_train,verbose = 2 ,epochs = 25)
test_loss,test_accuracy = model.evaluate(x_test,y_test)

# for layer in model.layers :
#   layer.trainable = False

model.save('digits')              