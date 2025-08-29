import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder # for encoding categorical columns   
from sklearn.model_selection import train_test_split # for splitting the dataset
from xgboost import XGBRegressor # for XGBoost Regressor Model
from sklearn import metrics # for checking the accuracy of the model


mart_dataset = pd.read_csv('mart_sales.csv')
# print(mart_dataset.info) # (rows, columns)

mart_dataset['Item_Weight'].mean() # mean of the Item_Weight column

# replacing the null values in the Item_Weight column with the mean value of that column
mart_dataset['Item_Weight'] = mart_dataset['Item_Weight'].fillna(mart_dataset['Item_Weight'].mean())
# print(mart_dataset.isnull().sum()) # getting some info about the dataset

# replacing missing values in "Outlet_Size" with mode
# Get mode of Outlet_Size per Outlet_Type
mode_of_outlet_size = mart_dataset.pivot_table(
    values='Outlet_Size',
    index='Outlet_Type',
    aggfunc=lambda x: x.mode()[0]
)

# Convert to a mapping dictionary
mode_mapping = mode_of_outlet_size['Outlet_Size'].to_dict()

# Fill missing Outlet_Size values using Outlet_Type mapping
mart_dataset['Outlet_Size'] = mart_dataset['Outlet_Size'].fillna(
    mart_dataset['Outlet_Type'].map(mode_mapping)
)


# Numberical Features
sns.set()

# Item_weight distribution
plt.figure(figsize=(6,6))
sns.displot(mart_dataset['Item_Weight'])

# outlet establishment year column 
plt.figure(figsize=(6,6))
sns.countplot(x='Outlet_Establishment_Year', data=mart_dataset)
# plt.show()


# data refining 
item_fat = mart_dataset['Item_Fat_Content'].value_counts()
item_fat_replace = mart_dataset.replace({'Item_Fat_Content' : {'low fat':'Low Fat','LF':'Low Fat', 'reg':'Regular'}}, inplace=True)
item_fat = mart_dataset['Item_Fat_Content'].value_counts()
# print(item_fat)


# ... Label Encoding ...
# Label Encoding - Create separate encoders for each column and save them
encoders = {}
categorical_columns = ['Item_Identifier', 'Item_Fat_Content', 'Item_Type', 'Outlet_Identifier', 
                      'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type']

for col in categorical_columns:
    encoder = LabelEncoder()
    mart_dataset[col] = encoder.fit_transform(mart_dataset[col])
    encoders[col] = encoder  # Save the fitted encoder

# ------- splitting features and target ----------
X = mart_dataset.drop(columns='Item_Outlet_Sales', axis=1)
Y = mart_dataset['Item_Outlet_Sales']

# ------- splitting data into training data & test data --------
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# -------- training the model with XGBRegressor ----------
regressor = XGBRegressor()
regressor.fit(X_train, Y_train)

# -------- Evaluation -----------
# prediction on training data
r2_training_data_prediction = regressor.predict(X_train)
r2_train = metrics.r2_score(Y_train, r2_training_data_prediction)

# prediction on test data
r2_test_data_prediction = regressor.predict(X_test)
r2_test = metrics.r2_score(Y_test, r2_test_data_prediction)

print(f"Training R²: {r2_train}")
print(f"Test R²: {r2_test}")

# Prediction for new data
raw_input = input("Enter data for XGB model prediction: ")
input_values = raw_input.split(",")

# Convert numerical values
input_values[1] = float(input_values[1])  # Item_Weight
input_values[3] = float(input_values[3])  # Item_Visibility
input_values[5] = float(input_values[5])  # Item_MRP
input_values[7] = float(input_values[7])  # Outlet_Establishment_Year

# Create DataFrame
input_df = pd.DataFrame([input_values], columns=['Item_Identifier', 'Item_Weight', 'Item_Fat_Content', 
                                                'Item_Visibility', 'Item_Type', 'Item_MRP', 
                                                'Outlet_Identifier', 'Outlet_Establishment_Year', 
                                                'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type'])

# Apply the same encoding as during training
for col in categorical_columns:
    # For new values not seen during training, we need to handle them
    try:
        input_df[col] = encoders[col].transform(input_df[col])
    except ValueError:
        # If value wasn't seen during training, assign a default value (like -1)
        # Or you could use the most frequent category
        print(f"Warning: New category '{input_df[col].iloc[0]}' in column '{col}'")
        input_df[col] = -1  # Or handle differently

# Ensure all columns are numeric
input_df = input_df.astype(float)

# Make prediction
prediction = regressor.predict(input_df)
print(f'Predicted Sales: {prediction[0]}')
print('Input data after encoding:')
print(input_df)


# ------- Training the model with Support Vector Machine (SVM) for REGRESSION ----------
from sklearn.svm import SVR # for Support Vector Machine Model
from sklearn import metrics # for checking the accuracy of the model

classifier = SVR(kernel='linear')
classifier.fit(X_train, Y_train)

# prediction on training data
r2_training_data_prediction = classifier.predict(X_train)

# Use REGRESSION metrics, not accuracy_score
r2_train = metrics.r2_score(Y_train, r2_training_data_prediction)
print(f'Accuracy on r2 training data: {r2_train}')

# prediction on test data
r2_test_data_prediction = classifier.predict(X_test)
r2_test = metrics.r2_score(Y_test, r2_test_data_prediction)
print(f'Accuracy on r2 test data: {r2_test}')

svr_prediction = classifier.predict(input_df)
print(f'SVR Predicted Sales: {svr_prediction[0]}')
print('Input data after encoding:')
print(input_df)
