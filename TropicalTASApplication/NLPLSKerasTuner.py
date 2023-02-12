#!/usr/bin/env python3

# import statements
import numpy as np
import xarray as xr
import glob
import PolarTestingTrainingSplit_CV
import tensorflow as tf
from tensorflow import keras
import keras_tuner
from keras_tuner import BayesianOptimization
from sklearn.cross_decomposition import PLSRegression
from numpy.random import seed
seed(0)
tf.random.set_seed(0)

# Get names of models in which we are testing on
path_to_data = '/home/disk/pna2/aodhan/SurfaceTrendLearning/PoChedleyEtAl2022/TASmaps/*_TrendMaps.nc'
ModelNames = [i[70:-16] for i in glob.glob(path_to_data)]

# Do train-test-split 
TrainingPredictorData, TrainingTargetData, TestingPredictorData, TestingTargetData, TestingTotalTrend = PolarTestingTrainingSplit_CV.training_testing_split(path_to_data)

# Store summaries
summaries = []

# Define the model
def build_model(hp):
    model = keras.Sequential()
    model.add(keras.layers.Dense(units=hp.Int('units', min_value=2, max_value=32, step=2), activation='relu', input_shape=(6,)))
    model.add(keras.layers.Dense(units=1))
    model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', [1e-4,1e-5])), loss='mean_squared_error')
    return model

# Loop over all cross validations
for model_idx in range(0, len(ModelNames)):
    
    # Define the tuner
    tuner = BayesianOptimization(build_model, objective='val_loss', max_trials=8, overwrite=True, 
    directory='/home/disk/p/aodhan/SurfaceTrendLearing/TropicalTASApplication/output',
    project_name="NLPLS_1Layer_6components_{model}".format(model=ModelNames[model_idx]), seed=0)
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

    # find specific model cross validation data
    TrainingTargetDataShape = np.shape(TrainingTargetData[model_idx])
    TestinTargetDataShape = np.shape(TestingTargetData[model_idx])
    TrainingTargetDataReshaped = np.reshape(TrainingTargetData[model_idx], (TrainingTargetDataShape[0], TrainingTargetDataShape[1]*TrainingTargetDataShape[2]))
    TestingTargetDataReshaped = np.reshape(TestingTargetData[model_idx], (TestinTargetDataShape[0], TestinTargetDataShape[1]*TestinTargetDataShape[2]))
    TrainingTargetDataReshaped = np.transpose(TrainingTargetDataReshaped[:,0])
    TestingTargetDataReshaped = np.transpose(TestingTargetDataReshaped[:,0])

    # Model Design
    pls = PLSRegression(n_components=6)
    
    # Use PLS regression to find reduced space
    pls_model = pls.fit(TrainingPredictorData[model_idx], TrainingTargetDataReshaped)
    X_train_pls = pls.transform(TrainingPredictorData[model_idx])
    X_test_pls = pls.transform(TestingPredictorData[model_idx])
    
    # Set training and testing data
    X_train =X_train_pls# np.reshape(TrainingPredictorData[model_idx], (len(TrainingPredictorData[model_idx]),72*144))
    X_test = X_test_pls#np.reshape(TestingPredictorData[model_idx], (len(TestingPredictorData[model_idx]),72*144))
    Y_train = TrainingTargetDataReshaped
    Y_test = TestingTargetDataReshaped

    # Fit the model using separate training and validation data
    history = tuner.search(X_train, Y_train, epochs=250, validation_data=(X_test, Y_test), 
                           verbose=0, callbacks=[stop_early])

    # Get best model
    best_model = tuner.get_best_models()[0]
    
    # evaluate best model
    val_loss = best_model.evaluate(X_test, Y_test, verbose=0)

    # save best hyper parameters and save them
    best_hps = tuner.get_best_hyperparameters()[0]
    summaries.append([ModelNames[model_idx], best_hps.get('units'), best_hps.get('learning_rate'), val_loss])
    
    # Print preformance and optimal hyper parameters
    print(ModelNames[model_idx])
    print('MSE:', val_loss)
    print('Optimal HPs: ', best_hps.get('units'), best_hps.get('learning_rate'))
    print('__________________________________________________')

# save preformence and hyperparameters to a separate text file
np.save("NLPLS_1Layer_6components.npy", summaries)