import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

# Load dataset
data = pd.read_csv("diabetes.csv")

# Features (X) and target (y)
X = data.drop(columns="Outcome", axis=1)
y = data["Outcome"]

# Standardize
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Train
model = LogisticRegression(max_iter=200)
model.fit(X_train, Y_train)

# Example patient data
input_data = (3,126,88,41,235,39.3,0.704,27)  # 8 values, same order as dataset columns

# Convert to numpy and reshape (1 row, 8 features)
input_array = np.asarray(input_data).reshape(1, -1)

# Standardize this input using the SAME scaler
std_input = scaler.transform(input_array)

# Predict
prediction = model.predict(std_input)
print(prediction)

print("Prediction:", prediction)  # [0] or [1]
if prediction[0] == 1:
    print("The person is diabetic")
else:
    print("The person is NOT diabetic")
