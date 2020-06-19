#%%
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import pickle




# %%




X = pickle.load(open("data/kaggle_cats_and_dogs/processed_data/X.pickle", "rb"))
y = pickle.load(open("data/kaggle_cats_and_dogs/processed_data/y.pickle", "rb"))

X = X / 255

#%%
dense_layers = [0, 1, 2]
layer_sizes = [32, 64, 128]
conv_layers = [1, 2, 3]

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            NAME = f"{conv_layer}-conv-{layer_size}-nodes-{dense_layer}-dnes-{int(time.time())}"
            print(NAME)

            tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

            model = Sequential()
            model.add(Conv2D(64, (3, 3), input_shape=X.shape[1:]))
            model.add(Activation("relu"))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            for l in range(conv_layer-1):
                model.add(Conv2D(64, (3, 3)))
                model.add(Activation("relu"))
                model.add(MaxPooling2D(pool_size=(2, 2)))
            
            model.add(Flatten())
            for l in range(dense_layer):
                model.add(Dense(layer_size))
                model.add(Activation("relu"))

            model.add(Dense(1))
            model.add(Activation('sigmoid'))

            model.compile(loss="binary_crossentropy",
                            optimizer="adam",
                            metrics=['accuracy'])

            model.fit(X, y, batch_size=32, validation_split=0.1, epochs=10, callbacks=[tensorboard])



# %%
model.save("cats_and_dogs_classifier.model")

# %%
