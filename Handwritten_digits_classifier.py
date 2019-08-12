#import main libraries
import tensorflow as tf
from tensorflow import keras
#helper libraries
import matplotlib.pyplot as plt
import numpy as np

#labels
categories = ["0","1","2","3","4","5","6","7","8","9"]

#load data set from mnist
number_mnist = tf.keras.datasets.mnist

#split into train and test
(train_images, train_labels), (test_images, test_labels) = number_mnist.load_data()

#normalise data - all values are between 0 and 1
train_images = train_images / 255.0
test_images = test_images / 255.0

#setup the layers of the model
model = keras.Sequential([
    #turns a 28x28 array into a 784x1 array
    keras.layers.Flatten(input_shape=(28, 28)),
    #helps reduce overfitting of data
    keras.layers.Dropout(0.2, noise_shape=None, seed=None),
    #add a layer to the neural net - outputs an array of length 128
    keras.layers.Dense(128, activation=tf.nn.relu),
    #final output layer returning an array of length 10 - maps all values to either a 0 or 1
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

#complie the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#training the model using train_images
model.fit(train_images, train_labels, epochs=5)

#run the model on the test data
test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)

#make predicitions for test images
predictions = model.predict(test_images)

print(predictions[0])
print(np.argmax(predictions[0]))
print(test_labels[0])
