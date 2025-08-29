# converting data to a numerical value 

import pandas as pd
import sklearn
from sklearn.preprocessing import LabelEncoder

iris_data = pd.read_csv('data/iris.csv', header=None)
iris_data.columns = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']

vc = iris_data['Species'].value_counts()

# encode label
label_encode = LabelEncoder()
labels = label_encode.fit_transform(iris_data.Species)

# append it to the iris data
iris_data['target'] = labels

# it will label can comes in alphabetical order
print(iris_data['target'].value_counts())


# 
# 
# ---- Feature Extraction of texts data using Tf-idf Vectorizer ----
# converting texts into ml output (numbers) where the machine understands

# importing libaries
# import numpy, pandas, mltk, re
# from mltk.corpus import stopwords
# from mltk.stem.porter import PoeterStemmer
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer

# stop words are words that are reoccuring | frequent in our data #
# mltk.download('stopwords')
# print(stopwords.words('english'))

# checking for missing values
    # news_data.isnull().sum()
# incase of numerical value we can replace the missing valuse as mean, median or mode values
# since it a text; we'll replace missing values with 'null' (string) empty string
    # news_data = news_data.fillna('')

# merging author name and news title together to make a new column 
    # news_data['content'] = news_data['author'] + ' ' + news_data['title']
    
# seperating data into feature and target
    # X = news_data.drop(columns='label', axis=1)
    # Y = news_data['label']

# Stemming: Stemming is a process of reducing a word to its Root word
    # port_stem = PorterStemmer()
    # def stemming(content):
        # stemmed_content = re.sub('[^a-zA-Z]','', content)
        # stemmed_content = stemmed_content.lower()
        # stemmed_content = stemmed_content.split()
        # stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
        # stemmed_content = ' '.join(stemmed_content)
        # return stemmed_content
        
    # news_data = news_data['content'].apply(stemming)

# converting the textual data to feature vectors
    # vectorize = TfidfVectorizer
    # vectorize.fit_transform(x)



# 
# 
# ---- STEPS IN DATA PROCESSING --- 
# 1. loading data from data file path

# 2. Handling Missing values if ANY
    # imputation 
    # droping

# 3. split data into features and target (target:- what you want to get | outcome e.g. end result, while features are the other values)
    # a. x = diabeties_data.drop(columns='name of column to exclude', axis=-1) 
        # this will mpve all featues in x and exclude the target column
    # b. y = diabeties_data['name of target column']
        # save only the target column in y 
        
# ALL IMPORTS HERE ---
# 4. Data Standardization :- this will help datas to be in similar | common range
    # a. import sklearn
        # from sklearn.preprocessing import StrandardScaler
        # from sklearn.model_selection import train_test_split
        # from sklearn.metrics import accuracy_score
        # from sklearn.model_selection import LogisticRegression
    # b. scaler = StandardScaler()
        # standardized_data = scaler.fit_transform(x)
    
# 5. Spliting the data into training and testing data
    # Create 4 arrays :- x_train, x_test, y_train, y_test
        # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
        

# 6a. Model Training => (Logistic Regression)
    # model = LogisticRegression()
        # - Training the logistic regression model with training data
            # model.fit(X_train, Y_train)
        
        # Model Evaluation :- accuracy of a model depends on the amout of data trained with it 
                            # any accuracy > 70%, is good
            
            # X_train_prediction = model.predict(X_train)
                # training_data_accuracy = accuracy_score(X_train, Y_train)
                    # print accuracy on training   data:- print(training_data_accuracy)       
                    # print accuracy on test       data:- print(test_data_accuracy)       
        
        # Making a Predictive System
            # input_data = ()
            # - changing the input_data to a numpy array
                # input_data_as_numpy_array = np.asarray(input_data)
            
            # - reshap the np array as we are predicting for one instance
                # input_data_reshaped = input_data_as_numpy_array.reshape(1, -1) 
                    # prediction = model.predict(input_data_reshaped)
                        # print(prediction)
                        
                        
                        
# 6b. Model Training => (SVM :- Support Vector Machine) ----- Diabeties Training Machine Model
# from sklearn import svm  
    # classifier = svm.SVC(kernel='linear')
        # - Training the vetor support model with training data
            # classifier.fit(X_train, Y_train)
        
        # Model Evaluation :- accuracy of a model depends on the amout of data trained with it 
                            # any accuracy > 70%, is good
            
            # X_train_prediction = classifier.predict(X_train)
                # training_data_accuracy = accuracy_score(X_train, Y_train)
                    # print accuracy on training   data:- print(training_data_accuracy)       
                    # print accuracy on test       data:- print(test_data_accuracy)       
        
        # Making a Predictive System
            # input_data = ( paste the value to test here... )
            # - changing the input_data to a numpy array
                # input_data_as_numpy_array = np.asarray(input_data)
            
            # - reshape the np array as we are predicting for one instance
                # input_data_reshaped = input_data_as_numpy_array.reshape(1, -1) 
                    # standardize the input data with (scaler) before prediction
                        # std_data = scaler.transform(input_data_reshaped)
                            # prediction = classifier.predict(std_data)
                                # print(prediction)
                                
            # Get on :- write a fn that print if the user has a diabetes ('Yes is diabetic') : ('nill')