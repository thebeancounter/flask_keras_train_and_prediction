from keras.datasets import mnist
from keras.layers import Dense, Dropout, Softmax, Flatten
from keras.losses import categorical_crossentropy
from keras.optimizers import SGD
from keras.models import Sequential
import keras

num_classes = 10

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


model = Sequential()
model.add(Dense(50, activation="relu", input_shape=x_train.shape[1:]))
model.add(Dense(10, activation="softmax"))
model.compile(loss=categorical_crossentropy, optimizer=SGD(), metrics=["accuracy"])

model.fit(x_train, y_train, epochs=10)