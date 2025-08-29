# The purpose of this code snipp is to determine the accuracy of each models, using some models

# comparing the models: list of models
# models = [LogisticsRegression(), SVC(kernel='linear',), KNeighborsClassifier(), RandomForestClassifier()]

# looping through the models to get each model
    # for model in models:
    #  training the model
        # model.fit(X_train, Y_train)
    #  evaluating the model 
        # test_data_prediction = model.predict(X_test)
    #  model accuracy
        # accuracy = accuracy_score(Y_test, test_data_prediction)
    # print('Accuracy score for, model, '=' accuracy) 
    
    
# In this code snipp, we are using cross validation instead of Test_Split_Data, using LogisticRegression Model
# cv_score_lr = cross_val_score(LogisticRegression(max_iter= 1000), X, Y cv=5)
# Note: add the score together then divide by the lenght (just like finding the MEAN val)
# mean_accuracy_lr = sum(cv_score_lr)/len(cv_score_lr)
# mean_accuracy_lr = mean_accuracy_lr*100
# mean_accuracy_lr = round(mean_accuracy_lr, 2)
# print(mean_accuracy_lr)

