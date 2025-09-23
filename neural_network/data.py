
# Neural Networks 
# input layer => hidden layer => output layer
# data goes in => leanrs pattern => output learned predictions | probabilities


# HOW IT WORKS :- 
# input => numeral encoding => learns representation(patters | features | weights) => 
# representaion output (numerical output) => user output (classification | continuous values)

# TENSORFLOW (IN VIEW)
# uses GPU :- this breaks numerical values. or TPU:- Tensor Processing unit
# The numerical values (standardlize digits) stands as Tensors 

# STEPS in modelign with Tensorflow - - -
# 1. creating a model - define the input and output layer, as well as the hidden layers of a deep learning model
# 2. compiling a model - define the loss function (the funtion which tells our model how wrong it is) and the optimizer
#    (tell our model how to improve patterns its learning) and evaluation metrics (what we can use to interprete t6he performance of our model) 
# 3. fitting a model - letting the model try to find patterns between X & Y (features and labels) 

# Improving our model 
# 1. creating a model - here we might add more layers, increase the number of hidden uints (all called neurons) which within 
#    each of the hidden layers, change the activation function of each layer. 
# 2. compiling a model - here we might change the optimizer function or perhaps the LEARNING RATE of the optimization function 
# 3. Fitting a model - here we might fit a model for more **epochs** (leaving it training for longer) or on more data
# 4. Make the model larger - (using a more complex model) this might come in form of more layers or more hidden units in each layer

# Model Evaluation 
# For Regression Model:- 
    # MAE :- mean absolute error, 'on average, how wrong is each of the model's prediction'
    # MSE :- mean square error, 'square the average errors'
    # Huber :- combination of MSE and MAE
    
# Saving Models:- There are two formats 
# The savedModel Format :- modelName.save("model_saving_name")
# The HDF5 Format :- modelname.save('model_name.h5')

# Loading saved models:-
# loaded_savedModel_format = tf.keras.models.load_model('model_path')
# confirm:- loaded_savedModel_format.summary()

# Downloading model from google colab
# from google.colab import files
# files.download('path_to_file')