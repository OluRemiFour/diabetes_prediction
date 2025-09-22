import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# create tensor with tf.constant()
scalar = tf.constant(7)

# checking the dimension of a tensor (ndim stands for number of dimension)
# print(scalar.ndim)

# create a vector
vector = tf.constant([10, 10])
# print(vector)

# create a matrix
matrix = tf.constant([[10, 8],
                      [2, 4]])

# create a valid 3D tensor
tensor = tf.constant(
    [
        [
            [[1, 2, 3], [4, 5, 6]],
            [[7, 8, 9], [10, 11, 12]]
        ],
        [
            [[13, 14, 15], [16, 17, 18]],
            [[19, 20, 21], [22, 23, 24]]
        ]
    ]
)

# Scalar : a single number 
# Vector : a number with direction (e.q. wind speed and directon)
# Matrix : a 2 dimentional array of numbers
# Tensor : an n-dimentional array of numbers (when n can be any number, a 0-dimentional tensor is a scalar, a 1-dimentional tensor is a vector)
 
# Excercise
# read through Tensorflow docs on random seed generation and practice writing 5 random tensors and shuffle them

# create a tensors of all ones
tf.ones([10, 7])

# create a tensors of all zeros
tf.zeros(shape=(4, 9))


# ---- Turn NumPy arrays into Tensors
    # The main diff between NumPy arrays and TensorFlow tensors is that tensors can run on a GPU (much faster for numerical computing)

# numpy_array = np.arrang(1, 25, dtype=np.int32)    # create an array between 1 and 25

# convert to tensors
# A = tf.constant(numpy_array, shape(2, 3, 4))

# Excercise:- play around arrays and adjust the shapes to fit into diff size  
arrs = tf.constant(np.random.randint(2, 8, size=10))

# Indexing tensors -- - --
# get the first 2 element of each dimension
# element_indexed[:2, :2, :2]

# work on tf.trasnspose

# Changing the datatype of a tensor

# create a ranadom tensor with values betw 0 an d 100 of size 50
E = tf.constant(np.random.randint(0, 100, size=50))
# print(E)

# Find the minimum of a tensor
min_tensor = tf.reduce_min(E)
# print(min_tensor)

# One Hot Encoding Tensors -- =>
some_list = [0, 1, 2, 3]
encode = tf.one_hot(some_list, depth=4)
# print(encode)

# Finding access to GPUS
mem = tf.config.list_physical_devices()
# print(mem)

# ----- => ----- => ------ => ------ => ------
# Demo tensor for housing price prediction problem 

house_info = tf.constant(['bedroom', 'bathroom', 'garage'])
house_price = tf.constant([939700])

x = np.array([-7.0, -4.0, -1.0, 2.0, 5.0, 8.0, 11.0, 14.0])
y = np.array([3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0])

# Turn NumPy arrays into tensors
x = tf.constant(x)
y = tf.constant(y)

x = tf.reshape(x, (-1, 1))  # reshapes tensor into (8, 1)

# setting up the model - - - -
# set random seed
tf.random.set_seed(42)

# model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(100, activation='relu'),
    # tf.keras.layers.Dense(100, activation='relu'),
    # tf.keras.layers.Dense(100, activation='relu'),
    # tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(1)
])  

# compile the model
model.compile(loss=tf.keras.losses.mae,     # mae is short for mean absolute error
             optimizer = tf.keras.optimizers.Adam(lr=0.01), # Adam | sgd  # sgd is stochastic gradient descent (how neural network should improve)
             metrics = ['mae'])  # human interpretable values for how well your model is doing  

# fit the model
model.fit(x, y, epochs=100)

# predict
y_pred = model.predict([17.0])
# print(y_pred)

X_train = x[:40]
Y_train = y[:40]

X_test = x[40:]
Y_test = y[40:]


print(x[0])
print(X_train[0])