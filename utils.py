nb_classes = 10

def get_data():
    from keras.datasets import mnist
    from keras.utils import np_utils
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(60000, 784)
    X_test = X_test.reshape(10000, 784)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    return (X_train, Y_train), (X_test, Y_test)

def build_model(layers = 2, size = 512, dropout = None, **kwargs):
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Input
    model = Sequential()

    for i in range(layers):
        if i:
            model.add(Dense(size, activation="relu"))
        else:
            model.add(Dense(size, activation="relu", input_shape=(784,)))

        if dropout:
            model.add(Dropout(0.2))


    model.add(Dense(10, activation="softmax"))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
    return model


def model_without_dropout():
    from keras.models import Sequential
    from keras.layers import Dense, Dropout
    model = Sequential()
    model.add(Dense(512, input_shape=(784,), activation="relu"))
    model.add(Dense(512, activation="relu"))
    model.add(Dense(10, activation="softmax"))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
    return model


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = get_data()
    model1, model2 = build_model(dropout=0.2), build_model(layers=4, size = 10, dropout=None)
    #model1.fit(x_train, y_train, batch_size=128, epochs=1, validation_data=(x_test, y_test), verbose=1)
    model2.fit(x_train, y_train, batch_size=128, epochs=100, validation_data=(x_test, y_test), verbose=1)