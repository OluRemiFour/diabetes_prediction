# model creation 
    # tweak around the activation='sigmoid'

# compilation :-
    # tweak around loss =, optimizers & learning rate   
    # metrics=['accuracy]

# after setup and compilation, evaluate the model with test dataset 

## Finding the best learning rate
    # To find the ideal learning rate where the loss decrease the most during training
        # A learning rate **callback**
            # after model setup creation & compilation . . .
            # lr_scheduler = tf.kares.callbacks.LearningRateScheduler(lambda epoch: le-4 * 10 ** (epoch/20))
            # then fit model = model_name.fit(X_train, Y_train, epoch=200, callbacks=[lr_schedular])

# Classification evaluation methods (read ...)
    # accuracy
    # precision
    # recall
    # f1-score
    # confusion-matrix


# Research: 
    # What change the values of the layers (layers.Dense(8, ...))