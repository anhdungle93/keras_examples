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

model = tf.keras.models.load_model("cats_and_dogs_classifier.model")
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
a = [1,2,3,4,5,6,7]

# %%
index = 0
while index <= len(a):
    print('index', index)
    print(len(a))
    if a is not None:
        del a[0]
    index += 1

# %%
