#%%
import cv2
import tensorflow as tf

#%%
CATEGORIES = ["Dog", "Cat"]

def prepare(filepath):
    IMG_SIZE = 75
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
#%%
model = tf.keras.models.load_model("cats_and_dogs_classifier.model")
#%%
print(model.summary())
#%%
prediction = model.predict([prepare('dog.jpg')])

# %%
prediction

# %%
prediction = model.predict([prepare('cat.jpg')])


# %%
prediction

# %%
prediction = model.predict_proba([prepare('landscape.jpg')])


# %%
prediction[0]

# %%
prediction[0][0]

# %%
for layer in model.layers:
    print(layer.name)
# %%

#%%

# %%
from tensorflow.keras import Model
intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer("dense_5").output)

# %%
model.input

# %%
intermediate_output = intermediate_layer_model([prepare('cat.jpg')])

# %%
import numpy as np
np.array(intermediate_output)

# %%
model.layers

# %%
print(model.summary())

# %%
model.layers[4]

# %%
