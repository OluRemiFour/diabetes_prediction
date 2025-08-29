import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score # for cross validation 
from sklearn.model_selection import GridSearchCV # for hyperparameter tuning 
from sklearn.metrics import accuracy_score # for checking the accuracy of the model

# importing models
from sklearn.linear_model import LogisticRegression # for Logistic Regression Model
from sklearn.svm import SVC # for Support Vector Machine Model
from sklearn.neighbors import KNeighborsClassifier # for K Nearest Neighbors Model
from sklearn.ensemble import RandomForestClassifier # for Random Forest Classifier Model

# loading the dataset to a Pandas DataFrame
heart_dataset =pd.read_csv('heart.csv')
# print(heart_dataset.shape) # (rows, columns)
# print(heart_dataset.head()) # first 5 rows of the dataset
# print(heart_dataset.isnull().sum()) # getting some info about the dataset
target_count = heart_dataset['target'].value_counts() # 1 --> Defective Heart, 0 --> Healthy Heart
# print(target_count)

# split into features & target
X = heart_dataset.drop(columns='target', axis=1)
Y = heart_dataset['target']

X = np.array(X)
Y = np.array(Y)

# print(X)
# print(Y)

# Model Selection & Comparing the Models with default hyperparameters values using Cross Validation
# The purpose of this code snipp is to determine the accuracy of each models, using some models
# comparing the models: list of models
models = [LogisticRegression(max_iter=10000), SVC(kernel='linear',), KNeighborsClassifier(), RandomForestClassifier()]

for model in models:
    cv_score = cross_val_score(model, X, Y, cv=5) # cv = number of folds
    mean_accuracy = sum(cv_score)/len(cv_score)
    mean_accuracy = mean_accuracy * 100
    mean_accuracy = round(mean_accuracy, 2)
    
    # print(f'Cross Validation Accuracyies score for {model} = {cv_score}')
    # print(f'Accuracy score fro {model} = {mean_accuracy} %')



# Model Selection & Comparing the Models with default hyperparameters values using GirdSearchCV
# The purpose of this code snipp is to determine the accuracy of each models, using some models
# comparing the models: list of models with hyperparameters values
model_list = [LogisticRegression(max_iter=1000), SVC(), KNeighborsClassifier(), RandomForestClassifier()]

# creating a dictionary of hyperparameters for each model
model_hyperparameters = {
    "log_reg_hyperparameters": {
        "C": [1, 5, 10, 20],
    },
    
    "svc_hyperparameters": {
        "C": [1, 5, 10, 20],
        "kernel": ['linear', 'rbf', 'poly', 'sigmoid']
    },
    
    "knn_hyperparameters": {
        "n_neighbors": [3, 5, 7, 9],
    },
    
    "random_forest_hyperparameters": {
        "n_estimators": [10, 20, 50, 100]
    }
}

print(model_hyperparameters.keys())
model_keys = list(model_hyperparameters.keys())

# Applying GridSearchCV
def ModelSelection(list_of_models, hyperparameters_dictonary):
    result = []
    i = 0
    for model in list_of_models:
        key = model_keys[i]
        params = hyperparameters_dictonary[key]
        i ++ 1

        classifier = GridSearchCV(model, params, cv=5)
        
        # fitting the model to the training data
        classifier.fit(X, Y)
        
        result.append({
            "model used": model,
            "highest score": classifier.best_score_,
            "best hyperparameters": classifier.best_params_
        })
        
        result_dataframe = pd.DataFrame(result, columns=["model used", "highest score", "best hyperparameters"])
        return result_dataframe