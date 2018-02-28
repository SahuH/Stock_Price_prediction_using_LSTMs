from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler
from keras.callbacks import Callback
import math
import time
import os
import warnings
import numpy as np
from numpy import newaxis
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from keras.utils.vis_utils import plot_model
from pandas import read_csv

class ResetStatesCallback(Callback):
	def __init__(self):
		self.counter = 0
		
	def on_batch_begin(self, batch, logs={}):
		self.model.reset_states()


class Lstm_btp(object):	
	def __init__(self, sequence_length = 2, epochs = 500, batch_size = 50, stateful = False, lr = 0.001):
		print('initialising class variables')
		self.fig_path = os.path.join(os.path.dirname(__file__), 'plots')
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
		
	def prepare_input(self, filename):
		data_df = read_csv(filename)
		data = data_df['Close'].values
		data1 = []
		for s in data:
			try:
				data1.append(float(s))
			except:
				continue
		data = data1
		data = np.array(data)
		#print(data.shape[0])
		data = self.difference(data,1)
		#print(data.shape[0])
		
		data = data.reshape(data.shape[0], 1)
		scaler = MinMaxScaler(feature_range=(-1, 1))
		scaler = scaler.fit(data)
		data = scaler.transform(data)
		data = data.reshape(data.shape[0],)
		#print(data.shape)
		
		seq_len = self.sequence_length + 1
		transformed_data = []
		for index in range(len(data) - seq_len):
			transformed_data.append(data[index: index + seq_len])
			
		transformed_data = np.array(transformed_data)
		
		for i in range(transformed_data.shape[0]):
			if transformed_data[i][-1] >= transformed_data[i][-2]:
				transformed_data[i][-1] = 1
			else:
				transformed_data[i][-1] = 0
		#np.random.shuffle(transformed_data)
		
		ratio = round(0.9 * transformed_data.shape[0])
		train = transformed_data[:int(ratio), :]
		#print(train[:10])
		self.X_train = train[:,:-1]
		self.y_train = train[:,-1]
		self.X_test = transformed_data[int(ratio):,:-1]
		self.y_test = transformed_data[int(ratio):,-1]
		
		
		self.vectorise_output()

		self.X_train = np.reshape(self.X_train, (self.X_train.shape[0], 1, self.X_train.shape[1]))
		self.X_test = np.reshape(self.X_test, (self.X_test.shape[0], 1, self.X_test.shape[1]))
		
	def difference(self, data, interval=1):
		diff = []
		for i in range(interval, len(data)):
			value = data[i] - data[i - interval]
			diff.append(value)
		return np.array(diff)
	

	def vectorise_output(self):
		self.y_train_vector = np.zeros((len(self.y_train), self.output_dim), dtype=np.float64)
		self.y_test_vector = np.zeros((len(self.y_test), self.output_dim), dtype=np.float64)
		label_indices_list_train = []
		label_indices_list_test = []

		for i in range(0,self.num_labels):
			label_indices_list_train.append([index for index,value in enumerate(self.y_train) if value in [self.class_var[i]]])
		for j in range(0,self.num_labels):
			label_indices_list_test.append([index for index,value in enumerate(self.y_test) if value in [self.class_var[j]]])
		
		
		for k in range(0,self.num_labels):
			for t in label_indices_list_train[k]:
				temp = np.zeros(self.num_labels)
				temp[k] = 1
				self.y_train_vector[t, :] = temp
				
		for l in range(0,self.num_labels):
			for u in label_indices_list_test[l]:
				temp = np.zeros(self.num_labels)
				temp[l] = 1
				self.y_test_vector[u, :] = temp	

		print("output vectors created for training...")

	def step_decay(self, epoch):
		initial_lr = 0.1
		drop = 0.5
		epochs_drop = 10.0
		lr = initial_lr * math.pow(drop, math.floor((1+epoch)/epochs_drop))
		return lr
		
	def define_model(self):
		model = Sequential()
		model.add(LSTM(batch_input_shape=(None, self.X_train.shape[1], self.X_train.shape[2]), units=5, stateful=self.stateful))
		model.add(Dropout(0.2))
		model.add(Dense(units=1))
		model.add(Activation('sigmoid'))
		sgd = SGD(lr=self.lr, momentum=0.9, decay=0.0, nesterov=False)
		model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
		self.model = model
		
	def learn_model(self):		
		if self.stateful:
			callbacks_list = [ResetStatesCallback()]
			self.history = self.model.fit(self.X_train, self.y_train, validation_data=(self.X_test, self.y_test), epochs=self.epochs, batch_size=1, callbacks=callbacks_list, shuffle=False)
			
		else:
			#lr = LearningRateScheduler(self.step_decay, verbose=1)
			#callbacks_list = [lr]
			self.history = self.model.fit(self.X_train, self.y_train, validation_data=(self.X_test, self.y_test), epochs=self.epochs, batch_size=self.batch_size, shuffle=False)
		
	def predict(self):
		self.y_predict_vector = self.model.predict(self.X_test, batch_size=1)
		scores = self.model.evaluate(self.X_test, self.y_test, verbose=0, batch_size=1)
		print(self.y_predict_vector[:20])
		print("Accuracy: %.2f%%" % (scores[1]*100))
	
if __name__ == '__main__':

	obj = Lstm_btp()
	obj.prepare_input('AAPL.csv')
	obj.define_model()
	obj.learn_model()
	plt.plot(obj.history.history['loss'], color='red')
	plt.plot(obj.history.history['val_loss'], color='blue')
	plt.xlabel('Epochs')
	plt.ylabel('Error')
	plt.legend(['Train', 'Validation'], loc='upper right')
	if obj.stateful:
		plt.title('stateful'+'  seq_len:'+str(obj.sequence_length)+'  l:'+'%.4f'%obj.history.history['loss'][-1]+'  a:'+'%.2f'%(obj.history.history['acc'][-1]*100)+
		'  val_l:'+'%.4f'%obj.history.history['val_loss'][-1]+'  val_a:'+'%.2f'%(obj.history.history['val_acc'][-1]*100)+' lr:'+str(obj.lr))
	else:
		plt.title('  seq_len:'+str(obj.sequence_length)+'  l:'+'%.4f'%obj.history.history['loss'][-1]+'  a:'+'%.2f'%(obj.history.history['acc'][-1]*100)+
		'  val_l:'+'%.4f'%obj.history.history['val_loss'][-1]+'  val_a:'+'%.2f'%(obj.history.history['val_acc'][-1]*100)+'  lr:'+str(obj.lr))
	w = 1
	plt.savefig(str(obj.fig_path)+'/image'+str(w)+'.png')
	plt.close()
	obj.predict()