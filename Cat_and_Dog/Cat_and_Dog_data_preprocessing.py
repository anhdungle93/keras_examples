#%%
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

DATADIR = "data/kaggle_cats_and_dogs/PetImages"
CATEGORIES = ["Dog", "Cat"]


# %%
IMG_SIZE = 75
training_data = []


# %%
def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass

create_training_data()
#%%
print(len(training_data))

# %%
import random

random.shuffle(training_data)

# %%

X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y = np.array(y)

# %%
import pickle

pickle_out = open("data/kaggle_cats_and_dogs/processed_data/X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("data/kaggle_cats_and_dogs/processed_data/y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()