
# converting to one hot encoding using .get_dummies(dataFrame)
# pd.get_dummies()



# using sklearn encoding
# from sklearn.compose import make_column_transformer
# from sklearn.preprocessing import MinMaxScaler, OneHotEncoding

# create a column transformer 
# read more on MinMaxScaler, Normalizing . . .
# ct = make_column_transformer(
            # (MinMaxScaler(), ['age', 'bmi', 'children'])          # turns all values in these columns between 0 and 1 
            # (OneHotEncoder(handle_unknown='ignore'),  ['sex', 'smoker', 'region'])
# )


# Transformer training and test data with normalization (MinMaxScaler) and OneHotEncoding 
# X_train_normal = ct.transform(X_train)
# X_test_normal = ct.transform(X_test)

# need more explanation on this . . . .
# fit the column transfomer to our training data
# ct.fit(X_train)

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



# -------------------
# classification NN
# read on: model:- ReLU (rectified linear unit), softmax activation, Sigmiod, compile:- loss fn (cross entropy)