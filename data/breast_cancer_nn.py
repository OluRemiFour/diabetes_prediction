import sklearn.datasets
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

breast_cancer_dataset = sklearn.datasets.load_breast_cancer() 
# print(breast_cancer_dataset)

data_frame = pd.DataFrame(breast_cancer_dataset.data, columns=breast_cancer_dataset.feature_names)

print(data_frame.columns.tolist())

X = data_frame.drop(columns=data_frame.target, axis=1)
Y = data_frame.target

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)


# standarlize the data 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.fit_transform(X_test)


# ----- importing tensorflow and Keras ------
import tensorflow as tf
tf.random.set_seed(3) # to get the same accuracy score 
from tensorflow import keras

# setting up the layers of Neural Network; where all the layers will be stacked
model = keras.Sequential([
                            keras.layers.Flatten(input_shape=(30)),
                            keras.layers.Dense(20, activation='relu'),
                            keras.layers.Dense(2, activation='sigmoid')                        
]) 

# compiling the Neural Netwotk 
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# training the Neural Network
history = model.fit(X_train, Y_train, validation_split=0.1, epochs=10)

# Accuracy of the model on test data
loss, accuracy = model.evaluate(X_test_std, Y_test)

# predicting the data; model.predict() gives the probability of each
# class for that data point
Y_pred = model.predict(X_test_std)

# converting the prediction probbility to calss labels
Y_pred_labels = [np.argmax(i) for i in Y_pred]
print(Y_pred_labels)

# building the predictive system
input_data = ()

# change the input_data to array 
input_data_as_array = np.asarray(input_data)

# reshape the numpy array as we are predictiing for one data point
input_data_reshaped = input_data_as_array.reshape(1, -1)

# standardizinf the input data
input_data_std = scaler.transform(input_data_reshaped)

# prediction from the model 
prediction = model.predict(input_data_std)
# print(prediction)

prediction_label = [np.argmax(prediction)]
# print(prediction_label)

if(prediction_label[0] == [0]):
    print('The tumoer is maglina')
    
else:
    print('The tumor is benigna')