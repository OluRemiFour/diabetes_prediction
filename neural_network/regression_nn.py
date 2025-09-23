import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pylab as plt 
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

insurance_dataset = pd.read_csv('insurance.csv')

# converting to one hot encoding using pandas.get_dummies()
# Convert categorical variable into dummy/indicator variables.
# data_scale = pd.get_dummies(insurance_dataset)
# print(data_scale)


# using sklearn encoding
# create a column transformer 
# read more on MinMaxScaler, Normalizing . . .
ct = make_column_transformer(
            (MinMaxScaler(), ['age', 'bmi', 'children']),          # turns all values in these columns between 0 and 1 
            (OneHotEncoder(handle_unknown='ignore'),  ['sex', 'smoker', 'region'])
)

X = insurance_dataset.drop(['charges'], axis=1)
Y = insurance_dataset['charges']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=42 ,test_size=0.2)

# fit the column transfomer to our training data
ct.fit(X_train)

# Transformer training and test data with normalization (MinMaxScaler) and OneHotEncoding 
X_train_normal = ct.transform(X_train)
X_test_normal = ct.transform(X_test)

# Model Compilation - - - -
# model = tf.kares.Sequential([
#     tf.kares.layers.Dense(100),
#     tf.kares.layers.Dense(10),
#     tf.kares.layers.Dense(1),
# ])

# model.compile(
#     loss=tf.kares.losses.mae,
#     optimizer=tf.kares.optimizers.SDG(),
#     metrics=['mae']
# )

# model.fit(X_train_normal, y_train, epochs=200, verbose=2)
# Evaluate the model
# model_name.evaluate(X_test_normal, Y_test)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(100, activation='softmax'),
    tf.keras.layers.Dense(10),
    tf.keras.layers.Dense(1)
])

model.compile(
    loss=tf.keras.losses.mae,
    optimizer=tf.keras.optimizers.SGD(),
    metrics=['mae']
)

model.fit(X_train_normal, Y_train, epochs=150, verbose=1)
model.evaluate(X_test_normal, Y_test)

Y_pred = model.predict(X_test_normal)
print(Y_test)

print('prediction ----')
print(Y_pred)

plt.figure(figsize= (10, 7))
plt.scatter(Y_test, Y_pred, c='g', label='Predicted vs True')
plt.xlabel('True values')
plt.ylabel('Predictions')
plt.legend()
plt.show()