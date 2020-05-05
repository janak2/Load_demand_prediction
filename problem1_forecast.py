#!/usr/bin/env python

'''
Author: Janak Agrawal
Date Created: March 06, 2020
Date Last Modified: March 06,2020
License: MIT
Python Version: 3.7
'''

import pandas as pd
import datetime as dt
from matplotlib import pyplot as plt
import tensorflow as tf
import os
import numpy as np
from tensorflow.keras.regularizers import l2

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


class ForecastModel:
	'''
	Class for creating a LSTM based time series prediction model.
	Attributes:
		TRAIN_SPLIT (int): The splitting index for train and test dataset
		HISTORY_SIZE (int): the number of data points needed to make prediction
		TARGET_SIZE (int): the number of data points to be predicted
	'''    
    
    def __init__(self,TRAIN_SPLIT,HISTORY_SIZE,TARGET_SIZE):
    	'''initialization function for the class'''

        self.TRAIN_SPLIT = TRAIN_SPLIT
        self.HISTORY_SIZE = HISTORY_SIZE
        self.TARGET_SIZE = TARGET_SIZE
    
    def normalize_data(self,features):
    	''' normalizes data around its mean and standard deviation'''

        dataset = features.values
        data_mean = dataset[:self.TRAIN_SPLIT].mean(axis=0)
        data_std = dataset[:self.TRAIN_SPLIT].std(axis=0)
        dataset = (dataset-data_mean)/data_std
        return dataset
    
    def prepare_labelled_data(self,dataset, target, start_index, end_index):
    	'''separate the dataset to create two datasets for training and label data points'''

        data = []
        labels = []

        #calculate start and end index for iteration
        start_index = start_index + self.HISTORY_SIZE
        if end_index is None:
            end_index = len(dataset) - self.TARGET_SIZE

        #aggregate data and append it to data and labels
        for i in range(start_index, end_index):
            indices = range(i-self.HISTORY_SIZE, i)
            data.append(dataset[indices])

            labels.append(target[i:i+self.TARGET_SIZE])

        return np.array(data), np.array(labels)
    
    def prepare_model(self,x_train):
    	'''implements and compiles the model architecture'''

    	#create a sequential model and add layers
        self.model = tf.keras.models.Sequential()
        
        self.model.add(tf.keras.layers.LSTM(32,
                                                  return_sequences=True,
                                                  input_shape=x_train.shape[-2:],activity_regularizer=l2(0.1)))
        
        self.model.add(tf.keras.layers.LSTM(16, activation='relu',activity_regularizer=l2(0.1)))
        self.model.add(tf.keras.layers.Dense(self.TARGET_SIZE,activity_regularizer=l2(0.1)))

        #compile model with optimizer
        self.model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mse',metrics=['mae', 'msle','mse'])
        
    def fit(self, train_data,val_data,EPOCHS,EVALUATION_INTERVAL):
    	'''trains the model to predict the train data'''

        self.model_history = self.model.fit(train_data, epochs=EPOCHS,
                                                  steps_per_epoch=EVALUATION_INTERVAL,
                                                  validation_data=val_data,
                                                  validation_steps=50)
        return self.model_history
    
    def predict(self,x):
    	'''makes a model prediction on data x'''

        return self.model.predict(x)
    
    def evaluate(self,x,y,BATCH_SIZE):
    	'''evaluates the loss of model based on data x and true prediction y'''

        return self.model.evaluate(x,y,batch_size=BATCH_SIZE)


class ForecastModelEvaluation:
	'''
	Class to evaluate the performance of ForecastModel class for training and test data.
	Attributes:
		model (ForecastModel) : the model to test performance
	'''
    
    def __init__(self,model):
    	'''initialization function for class'''
        self.model = model
        
    def evaluate(self,x_train,y_train,x_val,y_val,BATCH_SIZE=256):
    	'''evaluate the performance of model on train and test data'''

    	#call model evaluate function to get training losses
        losses =  self.model.evaluate(x_train,y_train,BATCH_SIZE)
        #get names of losses
        names = self.model.model.metrics_names

        #print losses on screen in order
        print("The losses were as follows for training(in-sample):")
        for i in range(1,len(losses)):
            print(names[i]+" : "+str(losses[i]))

    	#call model evaluate function to get test losses
        losses =  self.model.evaluate(x_val,y_val,BATCH_SIZE)

        #print losses on screen in order
        print("The losses were as follows for validation(out-sample):")
        for i in range(1,len(losses)):
            print(names[i]+" : "+str(losses[i]))


###----------------Data Cleansing-----------------###

#Read data using pandas
df = pd.read_csv('equity_RN628A_hbase_data.csv')
#print(df.head())

#Remove data that is not in between 11/02/2012 and 12/01/2013
df['dated'] = pd.to_datetime(df['date'])
start_date = dt.datetime(2012,11,2)
end_date = dt.datetime(2013,12,1)
df = df[df['dated']>=start_date]
df = df[df['dated']<=end_date]
#df.head()

#Fill in missing values using spline interpolation
print("Number of NaN in actual_kwh",df['actual_kwh'].isna().sum())
print("Number of NaN in temperature",df['actual_temperature'].isna().sum())

df['actual_kwh'].interpolate(method='spline',order=3,inplace=True)
df['actual_kwh'].fillna(method='bfill',inplace=True)
df['actual_temperature'].interpolate(method='spline',order=3,inplace=True)

print("Number of NaN in actual_kwh after interpolation",df['actual_kwh'].isna().sum())
print("Number of NaN in temperature after interpolation",df['actual_temperature'].isna().sum())

#Display how the data looks
plt.plot(df['actual_kwh'].head(200))
plt.show()
plt.plot(df['actual_temperature'].head(200))
plt.show()

#converting data into numpy arrays
features_name = ['actual_kwh','actual_temperature']
features = df[features_name]
features.index = df['Unnamed: 0']

###------------------------Model Data Preparation--------------------###

#Defining model parameters
TRAIN_SPLIT = int(len(df)*0.8) 
TARGET_SIZE = 4*24    #number of points to predict i.e. 24hrs
HISTORY_SIZE = 4*24*5 #number of points to use for prediction i.e. 5 days
TARGET_FEATURE = 0    #position of target feature in dataset
BATCH_SIZE = 512
EPOCHS = 10
EVALUATION_INTERVAL = 50
BUFFER_SIZE = 100000

#normalize and split dataset
kwh_model = ForecastModel(TRAIN_SPLIT,HISTORY_SIZE,TARGET_SIZE)
dataset = kwh_model.normalize_data(features)

x_train, y_train = kwh_model.prepare_labelled_data(dataset, dataset[:, TARGET_FEATURE], 0, TRAIN_SPLIT)
x_val, y_val = kwh_model.prepare_labelled_data(dataset, dataset[:, TARGET_FEATURE], TRAIN_SPLIT, None)

#Batch the data together
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

val_data = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_data = val_data.batch(BATCH_SIZE).repeat()


###--------------------Training model-------------------------------###

kwh_model.prepare_model(x_train)
history = kwh_model.fit(train_data,val_data,EPOCHS,EVALUATION_INTERVAL)

#make prediction using the model
for x, y in val_data.take(3):
    multi_step_plot(x[0], y[0], kwh_model.predict(x)[0])


###----------------Evaluating model-----------------------------###

kwh_evaluator = ForecastModelEvaluation(kwh_model)
kwh_evaluator.evaluate(x_train,y_train,x_val,y_val,BATCH_SIZE)



