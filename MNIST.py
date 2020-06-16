#%%
import tensorflow as tf

mnist = tf.keras.datasets.mnist  # 28

# %%
(x_train, y_train, x_test, y_test) = mnist.load_data()