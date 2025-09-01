import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 
from sklearn.linear_model import LogisticRegression

# Load dataset
breast_cancer_dataset = sklearn.datasets.load_breast_cancer()

# Create a DataFrame
data_frame = pd.DataFrame(breast_cancer_dataset.data, columns=breast_cancer_dataset.feature_names)

# Add target column
data_frame['label'] = breast_cancer_dataset.target

# Features and target
X = data_frame.drop(columns='label', axis=1)
Y = data_frame['label']

# checking distribution of target variable

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Model training
model = LogisticRegression(max_iter=10000)
model.fit(X_train, Y_train)

# Model evaluation
X_train_prediction = model.predict(X_train)

# Accuracy on training data
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print('Accuracy on training data : ', training_data_accuracy)

# Accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print('Accuracy on test data : ', test_data_accuracy)

input_data_prediction = input(f'input your prediction data here: ')

input_data_prediction_asList = [float(x) for x in input_data_prediction.split(',')]

# Change the input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data_prediction_asList)

# Reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# Predict result
prediction = model.predict(input_data_reshaped)
if(prediction == [0]):
    print('The Breast Cancer is Maligmant (M)')
else:
    print('The Breast Cancer is Benign (B)')