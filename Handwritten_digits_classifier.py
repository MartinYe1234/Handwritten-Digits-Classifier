#main libraries
import tensorflow as tf
from tensorflow import keras
#helper libraries
import matplotlib.pyplot as plt
import numpy as np
from random import randint

plt.figure()
plt.rcParams["figure.figsize"] = (15,15)

#function to help visualise results
def show_image():
  #generate a random batch of 6 consecutive cases to show to the user
  index = randint(0,9969)
  for i in range(6):
    cur_sample_index = index + i 
    #find most certain prediction
    most_certain = np.argmax(predictions[cur_sample_index])
    
    #show the actual written number being classified
    plt.subplot(3, 4, 2*i+1)
    #remove ticks from x and y axes
    plt.yticks([])
    plt.xticks([])
    plt.imshow(test_images[cur_sample_index])
 
    #print the results at the bottom of the graph
    pre_result = str(np.argmax(predictions[cur_sample_index]))
    ac_result = str(test_labels[cur_sample_index])
    #the certainty of the prediction
    certainty = str(round(100*np.max(predictions[cur_sample_index]),2))
    results = "Predicted: "+ pre_result + " (" + certainty  + "%)   Actual: "+ ac_result
    plt.xlabel(results)
    
    #show the certainty of the prediction next to the number
    plt.subplot(3, 4, 2*i+2)
    certainty_plot = plt.bar(range(10), predictions[cur_sample_index], color="grey")
    
    #set the line color depending on whether or not the prediction was correct
    if pre_result == ac_result:
      certainty_plot[most_certain].set_color("green")
    else:
      certainty_plot[most_certain].set_color("red")
    
    plt.ylim([0, 1])
    plt.xticks([i for i in range(11)])
  plt.show()

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
model.fit(train_images, train_labels, epochs=15)

#run the model on the test data
test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)

#make predicitions for test images
predictions = model.predict(test_images)

show_image()
