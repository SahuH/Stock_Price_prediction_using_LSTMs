''' implementation of model presented in rather paper as classification
'''
import time
import os, os.path
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler
from keras.callbacks import Callback
from keras.utils import np_utils
import math
import warnings
import numpy as np
from numpy import newaxis
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from math import sqrt
from keras.utils.vis_utils import plot_model
from pandas import read_csv

class ResetStatesCallback(Callback):
	def __init__(self):
		self.counter = 0
	def on_batch_begin(self, batch, logs={}):
		self.model.reset_states()

class Lstm_btp(object):	
	def __init__(self, sequence_length = 5, epochs = 50, batch_size = 512, stateful = False, lr = 0.1):
		print('initialising class variables')
		self.fig_path = os.path.join(os.path.dirname(__file__), 'plots_2')
		self.X_train = None 
		self.X_test = None 
		self.y_train = None
		self.y_test = None 
		self.y_train_vector = None
		self.y_test_vector = None
		self.y_predict_vector = None
		self.num_train_example = None
		self.num_test_example  = None
		self.y_train_vector = None
		self.y_test_vector = None
		self.num_features = None	
		self.class_var = [0, 1]
		self.num_labels = len(self.class_var) 
		self.output_dim = len(self.class_var)
		self.sequence_length = sequence_length
		self.epochs = epochs
		self.model = None
		self.history = None
		self.stateful = stateful
		self.lr = lr
		self.epochs_drop = 5
		self.batch_size = batch_size
		self.input_data = None
		self.preprocessed_data = None
		self.regressive_data = None
		
	def prepare_input(self, filename):
		data_df = read_csv(filename)
		data = data_df['Adj Close'].values
		data1 = []
		for s in data:
			try:
				data1.append(float(s))
			except:
				continue
		data = data1
		data = np.array(data)
		self.input_data = data
		'''
		data = data.reshape(data.shape[0], 1)
		scaler = MinMaxScaler(feature_range=(-1, 1))
		scaler = scaler.fit(data)
		data = scaler.transform(data)
		data = data.reshape(data.shape[0],)
		self.preprocessed_data = data
		'''		
		regressive_data = []
		seq_len = self.sequence_length
		y = []
		for index in range(len(data) - seq_len):
			regressive_data.append(data[index: index + seq_len])
			if (data[index+seq_len] - data[index+seq_len-1]) >= 0:
				y.append(1)
			else:
				y.append(0)			
			
		regressive_data = np.array(regressive_data)
		self.regressive_data = regressive_data		
		referenced_data = np.array(regressive_data)
		#print(transformed_data1.shape)
		for index in range(1,regressive_data.shape[0]):
			referenced_data[index] = regressive_data[index] - regressive_data[index-1,0]
			
		#print(regressive_data[:10])
		#print(referenced_data[:10])
		
		transformed_data = referenced_data[1:]
		y = y[1:]
		transformed_data1 = np.zeros((transformed_data.shape[0], transformed_data.shape[1]+1))
		transformed_data1[:,:-1] = transformed_data
		transformed_data1[:,-1] = y
		transformed_data = transformed_data1
		ratio = round(0.9 * transformed_data.shape[0])
		self.X_train = transformed_data[:int(ratio),:-1]
		self.y_train = transformed_data[:int(ratio),-1]
		self.X_test = transformed_data[int(ratio):,:-1]
		self.y_test = transformed_data[int(ratio):,-1]
		
		#self.y_train_vector = np_utils.to_categorical(self.y_train)
		#self.y_test_vector = np_utils.to_categorical(self.y_test)	

		print(self.X_train.shape, self.y_train.shape)
		print(self.X_test.shape, self.y_test.shape)		
		
	def inverse_reference(self, y):
		regressive_data = self.regressive_data
		y_inversed = []
		for i in range(y.shape[0]):
			y_inversed.append(y[i]+regressive_data[i,0])
		y_inversed = np.array(y_inversed)
		return y_inversed

	def define_model(self):
		self.X_train = np.reshape(self.X_train, (self.X_train.shape[0], self.X_train.shape[1], 1))
		self.X_test = np.reshape(self.X_test, (self.X_test.shape[0], self.X_test.shape[1], 1))	
		model = Sequential()
		if self.stateful:
			model.add(LSTM(batch_input_shape=(1, self.X_train.shape[1], self.X_train.shape[2]), units=5, stateful=self.stateful))
		else:
			model.add(LSTM(batch_input_shape=(None, self.X_train.shape[1], self.X_train.shape[2]), units=5, stateful=self.stateful))
		model.add(Dropout(0.2))
		model.add(Dense(units=1, activation='softmax'))
		model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
		self.model = model
		
	def learn_model(self):	
		if self.stateful:
			callbacks_list = [ResetStatesCallback()]
			self.history = self.model.fit(self.X_train, self.y_train, validation_data=(self.X_test, self.y_test), epochs=self.epochs, batch_size=1, callbacks=callbacks_list, shuffle=False)
			
		else:
			self.history = self.model.fit(self.X_train, self.y_train, validation_split=0.05, epochs=self.epochs, batch_size=self.batch_size, shuffle=False)
		
	def predict(self):
		X_test = self.X_test
		y_test = self.y_test
		y_predict = self.model.predict(X_test, batch_size=1)
		y_predict = y_predict.reshape(y_test.shape[0],)
		scores = self.model.evaluate(X_test, y_test, verbose=0, batch_size=1)
		print(y_predict[:20])
		print("Accuracy: %.2f%%" % (scores[1]*100))
		
	def plot_data(self):
		plt.scatter(self.X_train[:, 0], self.X_train[:, -1], marker='o', c=self.y_train)
		#plt.savefig('data distribution4.png')
		#plt.close()
		plt.show()
		
if __name__ == '__main__':
	obj = Lstm_btp()
	obj.prepare_input('AAPL_2.csv')
	#print(y1[:10])
	#print(obj.regressive_data[:10,-1])
	obj.plot_data()
	obj.define_model()
	obj.learn_model()
	obj.predict()