#%%
import cv2
import tensorflow as tf
import sys
sys.path.append('..')
import helping_script as hs
import numpy as np

#%%
CATEGORIES = ["Dog", "Cat"]

def prepare(filepath):
    IMG_SIZE = 75
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE)) / 255
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
#%%
model = tf.keras.models.load_model("cats_and_dogs_classifier.model")
#%%
_input = prepare('landscape.jpg')

# %%
outputs = hs.get_output_of_all_layers(model,_input)

# %%
print(model.summary())
# %%
outputs["activation_8"]

# %%
