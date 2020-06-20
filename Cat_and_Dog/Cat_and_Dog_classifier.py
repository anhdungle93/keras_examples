# %%
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import time
import pickle

NAME = "Cats-vs-dog-cnn-64x2-{}".format(int(time.time()))

tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

X = pickle.load(open("data/kaggle_cats_and_dogs/processed_data/X.pickle", "rb"))
y = pickle.load(open("data/kaggle_cats_and_dogs/processed_data/y.pickle", "rb"))

X = X / 255

model = Sequential()
model.add(Conv2D(64, (3, 3), input_shape=X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss="binary_crossentropy",
                optimizer="adam",
                metrics=['accuracy'])

model.fit(X, y, batch_size=32, validation_split=0.1, epochs=10, callbacks=[tensorboard])



# %%
model.save("cats_and_dogs_classifier.model")

# %%
