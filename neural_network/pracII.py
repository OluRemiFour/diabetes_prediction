import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

X = tf.range(-100, 100, 4)

Y = X + 10

X_train = X[:40]
Y_train = X[:40]

X_test = Y[40:]
Y_test = Y[40:]

X_train = tf.reshape(X_train, (-1, 1))


# Nueral Network
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, input_shape=[1], name='input_layer'),
    tf.keras.layers.Dense(1, name='output_layer')
], name='model_1')

model.compile(
    loss=tf.keras.losses.mae,
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=['mae']
)

model.fit(X_train, Y_train, epochs=100, verbose=0)  # verbose = 0 (dont show progress), verbose = 1 (show slight), verbose = 2 (show progress)
# model.summary()

plt.figure(figsize=(10, 7))
plt.scatter(X_train, Y_train, c='b', label='Training Data')
plt.scatter(X_test, Y_test, c='g', label='Test Data')
plt.legend()

# plot_model(model=model, show_shapes=True)

X_pred = model.predict(X_test)
# print('Prediction', X_pred)
# print(Y_test)

# Model Evaluation with regression evaluation metrics
# calculate the mean absolute error - -
mae = tf.metrics.mean_absolute_error(y_true=Y_test, y_pred=tf.squeeze(X_pred))
print(mae)


# calculate the mean square error 
mse = tf.metrics.mean_squared_error(y_true=Y_test, y_pred=tf.squeeze(X_pred))
print(mse)

# Experiment and improve the model by doing 3 experiments:
# 1. model_1 - same as the original model, 1 layer trained for 100 epochs
# 2. model_2 - 2 layers, trained for 100 epochs
# 2. model_3 - 2 layers, trained for 200 epochs

# set random seed
tf.random.set_seed(42)

# set up model - - --
model_1 = tf.keras.Sequential([
    tf.keras.layers.Dense(1)
])

model_1.compile(loss=tf.keras.losses.mae,
                optimizer=tf.keras.optimizers.SGD(),
                metrics=['mae'])

model_1.fit(X_train, Y_train, epochs=100)
Y_pred_1 = model_1.predict(X_test)

# calculate model_1 evaluation metrics - - - -
mae_1 = tf.metrics.mean_absolute_error(y_true=Y_test, y_pred=tf.squeeze(Y_pred_1))
mse_1 = tf.metrics.mean_squared_error(y_true=Y_test, y_pred=tf.squeeze(Y_pred_1))

plt.figure(figsize=(10, 7))
plt.scatter(X_train, Y_train, c='b', label='Training Data')
plt.scatter(X_test, Y_test, c='g', label='Testing Data')
plt.scatter(X_test, Y_pred_1, c='r', label='Predictions')
plt.show()


# SETUP:-  - - model_2 - -  
# model_2 = tf.keras.Sequential([
#     tf.keras.layers.Dense(10),
#     tf.keras.layers.Dense(1)
# ])

# model_2.compile(loss=tf.keras.losses.mae,
#                 optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
#                 metrics=['mae']    
# )

# model_2.fit(X_train, Y_train, epochs=200)
# Y_pred_2 = model_2.predict(X_test)

# plt.figure(figsize=(10, 7))
# plt.scatter(X_train, Y_train, c='b', label='Training Data' )
# plt.scatter(X_test, Y_test, c='g', label='Testing Data')
# plt.scatter(X_test, Y_pred_2, c='r', label='Prediction Result')
# plt.show()