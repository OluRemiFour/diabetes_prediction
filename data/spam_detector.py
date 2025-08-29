import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

mail_data = pd.read_csv("spam.csv", encoding="latin-1")

# Keep only the first two useful columns
mail_data = mail_data[["v1", "v2"]]

# Rename columns for clarity
mail_data.columns = ["category", "message"]

# extract features and target
X = mail_data["message"]
Y = mail_data["category"]

# Convert categories to numerical values
Y = Y.map({'ham': 0, 'spam': 1})


# Encode text data using TfidfVectorizer
vectorizer = TfidfVectorizer()

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

print(Y_train)


# Vectorize based on training data
feature_extraction = TfidfVectorizer(max_df=0.9, stop_words='english', lowercase=True)
x_train_feature = feature_extraction.fit_transform(X_train).toarray()
x_test_feature = feature_extraction.transform(X_test).toarray()

# Train the model using Logistic Regression
model = LogisticRegression(max_iter=200)
model.fit(x_train_feature, Y_train)

# Evaluate the model
predictions = model.predict(x_test_feature)
# input_data = ["Congratulations! You've won a lottery of $1000! Click here to claim your prize."]
input_data = ["Congratulations, fine wine is yours, to claim send won to 131"]
std_data = feature_extraction.transform(input_data).toarray()

predictions = model.predict(std_data)

print("Input Data:", input_data)
print("Predictions:", predictions)
# print("Accuracy:", accuracy_score(Y_test, predictions))
