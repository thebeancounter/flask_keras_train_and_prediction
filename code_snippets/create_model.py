from keras.layers import Dense, Dropout, Softmax, Flatten
from keras.losses import categorical_crossentropy
from keras.optimizers import SGD
from keras.models import Sequential

in_shape = (784,)
out_shape = 10

model = Sequential()
model.add(Dense(50, activation="relu", input_shape=in_shape)) #note, input_shape must be an iterable.
model.add(Dense(out_shape, activation="softmax"))
model.compile(loss=categorical_crossentropy, optimizer=SGD(), metrics=["accuracy"])