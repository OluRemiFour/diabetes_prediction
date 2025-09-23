from sklearn.datasets import make_circles
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np

n_samples = 1000
X, y = make_circles(n_samples, noise=0.03, random_state=42)

circles = pd.DataFrame({
    'X0': X[:, 0],   # first column of X
    'X1': X[:, 1],   # second column of X
    'label': y       # labels
})

# visualize with a plot
plt.scatter(
    X[:, 0], 
    X[:, 1], c=y, 
    cmap=plt.cm.RdYlBu)
# plt.legend()
# plt.show()

model_1 = tf.keras.Sequential([
    # tf.keras.layers.Dense(100, activation='relu'),  # 100 dense neurons
    tf.keras.layers.Dense(100),
    tf.keras.layers.Dense(1)  # 1 dense neurons
])

# model in classification specific
model_1.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                optimizer=tf.keras.optimizers.SGD(),
                metrics=['accuracy'])

# model_1.fit(X, y, epochs=50, verbose=1)

# plot decision boundary
def plot_decision_boundary(model, X, y):
    # Define the axis boundaries of the plot and create a meshgrid
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1  # <-- notice X[:,1]
    
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 100),
        np.linspace(y_min, y_max, 100)
    )
    
    # create X value (making predictions on these)
    x_in = np.c_[xx.ravel(), yy.ravel()]
    
    # make predictions
    y_pred = model.predict(x_in)
    
    # check for multi-class
    if len(y_pred[0]) > 1:
        print('doing multiclass classification')
        y_pred = np.argmax(y_pred, axis=1).reshape(xx.shape)
    else:
        print('doing binary classification')
        y_pred = np.round(y_pred).reshape(xx.shape)
    
    # plot the decision boundary
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(
        X[:, 0],
        X[:, 1],
        c=y,
        s=40,
        cmap=plt.cm.RdYlBu
    )
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())


# plot_decision_boundary(model=model_1, X=X, y=y)
# plt.show()


# creating regression problem . . .
tf.random.set_seed(42)

# create some regression data
X_regression = tf.range(0, 1000, 5)
Y_regression = tf.range(100, 1100, 5)

# split regression data into training and test data 
X_reg_train = X_regression[:150]
X_reg_test = X_regression[150:]

# reshape into (150, 1)
X_reg_train_reshape = np.array(X_reg_train).reshape(-1, 1)
X_reg_test_reshape = np.array(X_reg_test).reshape(-1, 1)

Y_reg_train = Y_regression[:150]
Y_reg_test = Y_regression[150:]



model_1.fit(X_reg_train_reshape, Y_reg_train, epochs=100, verbose=1)

# model in regression specific
tf.random.set_seed(42)

model_2 = tf.keras.Sequential([
    tf.keras.layers.Dense(100),
    tf.keras.layers.Dense(10),
    tf.keras.layers.Dense(1)
])

model_2.compile(loss=tf.keras.losses.mae,
                optimizer=tf.keras.optimizers.Adam(),
                metrics=['mae'])

model_2.fit(X_reg_train_reshape, Y_reg_train, epochs=100, verbose=1)
Y_model_2_pred = model_2.predict(X_reg_test)

plt.figure(figsize=(10, 7))
plt.scatter(X_reg_train_reshape, Y_reg_train, c='b', label='Training Data')
plt.scatter(X_reg_test_reshape, Y_reg_test, c='g', label='Testing Data')
plt.scatter(X_reg_test_reshape, Y_model_2_pred, c='r', label='Prediction')
plt.show()