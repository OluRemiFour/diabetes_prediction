# ------------ Formular for Linear Regression -----------------
# Y = mX + c
# where:-

# m => Slope (i.e. the line on that mark direction on the graph (bent, straignt or curvy))
# c => on the Y axis, the start (0) to where ht slope start 
# Y = Y value 
# X = X value

import numpy as np

class Linear_Regression: 
    def __init__(self, learning_rate, no_of_iteration):
        self.learning_rate = learning_rate
        self.no_of_iteration = no_of_iteration
            
    def fit (self, X, Y):
        # numbers of training examples & numbers of features
        self.m, self.n = X.shape  # numbers of column & rows        
        
        # initializing the weight and bias
        self.w = np.zeros(self.n)
        self.b = 0 
        self.X = X
        self.Y = Y
        
        # implement Gradient Descent
        for i in range(self.no_of_iteration):
            self.update_weights() 
            
    def update_weights(self):
        Y_prediction = self.predict(self.X)

        # calculating gradient 
        dw  = (2 * (self.X.T).dot(self.Y - Y_prediction)) / self.m
        db = 2 * np.sum(self.Y - Y_prediction) / self.m

        # updating the weight
        self.w = self.w - self.learning_rate * dw
        self.b = self.b - self.learning_rate * db

    
    def predict(self, X):
        return X.dot(self.w) + self.b 
     

# To use Linear Regression for Model
model = Linear_Regression(learning_rate=0.02, no_of_iteration=1000)
