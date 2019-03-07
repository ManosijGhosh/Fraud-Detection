import numpy as np
import signal
import sys
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from confusion import evaluate
import statistics


def create_model(ip_shape, lrate = 1e-5):

	#initialize data
	#X_test, X_test, Y_test = make_data()

	#initialize parameters
	model = dict()

	dim_input = ip_shape
	dim_enc_1 = 15
	dim_enc_2 = 8
	dim_dec_1 = 15
	dim_dec_2 = dim_input

	#create model
	with tf.name_scope('input'):
		model['ip'] = tf.placeholder(tf.float32, [None, dim_input], name = 'input-vector')
		model['labels'] = tf.placeholder(tf.float32, [1, None], name = 'input-vector')

	# FIRST ENCODING LAYER

	with tf.name_scope('encoding-layer-1'):
		model['We_1'] = tf.Variable(tf.random_normal([dim_input, dim_enc_1], stddev=1.0/dim_enc_1), name = 'We-1')
		model['Be_1'] = tf.Variable(tf.random_normal([1, dim_enc_1], stddev=1.0/dim_enc_1), name = 'Be-1')
		model['ye_1'] = tf.nn.tanh(tf.add(tf.matmul(model['ip'], model['We_1']), model['Be_1']), name = 'ye-1')

	# SECOND ENCODING LAYER

	with tf.name_scope('encoding-layer-2'):
		model['We_2'] = tf.Variable(tf.random_normal([dim_enc_1, dim_enc_2], stddev=1.0/dim_enc_2), name = 'We-2')
		model['Be_2'] = tf.Variable(tf.random_normal([1, dim_enc_2], stddev=1.0/dim_enc_2), name = 'Be-2')
		model['ye_2'] = tf.nn.tanh(tf.add(tf.matmul(model['ye_1'], model['We_2']), model['Be_2']), name = 'ye-2')

	# FIRST DECODING LAYER

	with tf.name_scope('decoding-layer-1'):
		model['Wd_1'] = tf.Variable(tf.random_normal([dim_enc_2, dim_dec_1], stddev=1.0/dim_dec_1), name = 'Wd-1')
		model['Bd_1'] = tf.Variable(tf.random_normal([1, dim_dec_1], stddev=1.0/dim_dec_1), name = 'Bd-1')
		model['yd_1'] = tf.nn.tanh(tf.add(tf.matmul(model['ye_2'], model['Wd_1']), model['Bd_1']), name = 'yd-1')
	
	# SECOND DECODING LAYER

	with tf.name_scope('decoding-layer-2'):
		model['Wd_2'] = tf.Variable(tf.random_normal([dim_dec_1, dim_dec_2], stddev=1.0/dim_dec_2), name = 'Wd-2')
		model['Bd_2'] = tf.Variable(tf.random_normal([1, dim_dec_2], stddev=1.0/dim_dec_2), name = 'Bd-2')

	with tf.name_scope('output'):
		model['op'] = tf.nn.sigmoid(tf.add(tf.matmul(model['yd_1'], model['Wd_2']), model['Bd_2']), name = 'output-vector')
		#model['op'] = tf.Print(model['op'], [model['op']], message="This is op: ", summarize=960)


	# loss calculation
	with tf.name_scope('loss_optim_4'):

		model['sq-diff'] = tf.reduce_mean(tf.squared_difference(model['ip'], model['op']),name='sq-diff')
		model['0-loss'] = tf.multiply(tf.subtract(tf.fill([1, tf.shape(model['labels'])[1]], 1.0),model['labels']),model['sq-diff'])
		model['1-loss'] = tf.multiply(model['labels'],tf.maximum(tf.subtract(model['labels'],model['sq-diff']),tf.fill([1,tf.shape(model['labels'])[1]],0.0)))
		model['cost'] = tf.add(model['0-loss'], model['1-loss'])
		model['optimizer'] = tf.train.AdamOptimizer(lrate).minimize(model['cost'], name = 'optim')

		model['cost-2'] = tf.reduce_mean(tf.squared_difference(model['ip'], model['op']), axis=1, name='cost-2')

	# return the model dictionary

	return model;

def train_model(model, X_train, X_labels, batchSize, path_model, path_logdir, epoch = 100):

	with tf.Session() as session:
		init = tf.global_variables_initializer()
		session.run(init)

		#path_model = './model-4'
		#path_logdir = 'logs-auto-4'

		saver = tf.train.Saver()
		writer = tf.summary.FileWriter(path_logdir, session.graph)

		for i in range(epoch):
			loss = np.empty([1,0])
			for count in range(0,X_train.shape[0],batchSize):
				in_vector = X_train[count:min(count+batchSize,X_train.shape[0])]
				in_labels = X_labels[count:min(count+batchSize,X_train.shape[0])]
				in_labels = np.asarray(in_labels).reshape(1,min(batchSize,X_train.shape[0]-count))
				#print(in_labels)
				#print(in_labels.shape)
				#in_labels = in_labels.transpose()
				#print(in_labels.shape())
				#in_vector = in_vector.reshape(1, in_vector.shape[0])
				feed = {model['ip']: in_vector, model['labels']: in_labels}

				_, summary = session.run([model['optimizer'], model['cost']], feed_dict = feed)
				loss = np.append(loss,summary)
				#print('for batch - ', (count//batchSize), ' - ', np.mean(summary), ' summary size - ',summary.shape)
				#print('Cost: ', model['cost'].eval())
			#writer.add_summary(summary, i) # AttributeError: 'numpy.ndarray' object has no attribute 'value'
			print('Epoch: ', i, 'loss - ',statistics.mean(loss))

		saver.save(session, path_model)

def test_model(model, X_test, batchSize, path_model, outfile):

	#path_model = './model-4'
	#path_logdir = 'logs-auto-2'

	with open(outfile, 'w+') as outfp:
		with tf.Session() as sess:

			saver = tf.train.Saver()
			saver.restore(sess, path_model)

			for i in range(0,X_test.shape[0],batchSize):

				in_vector = X_test[i:min(i+batchSize,X_test.shape[0])]
				feed = {model['ip']: in_vector}

				outfp.write('Batch --- ' + str(i//batchSize) + ' --- \n')
				ans = sess.run(model['cost-2'], feed_dict =  feed)

				outfp.write(str(list(ans)))
				outfp.write('\n')
		
def autoencoder(X_train, X_test, Y_train, str, fold):
	
	#X_test, X_train, Y_test = make_data()
	batchSize = 100
	
	'''
	negatives = list()
	for labels in range(0,X_train.shape[0]):
		if Y_train[labels]==1:
			negatives.append(labels)
	print(len(negatives))
	print('before - ',X_train.shape)
	X_train= np.delete(X_train, negatives, axis=0)
	'''
	print('after - ',X_train.shape)

	#'''
	lrate = 1e-5
	model = create_model(X_train.shape[1], lrate)
	train_model(model, X_train, Y_train, batchSize, path_model='./models/model_kfold_mano_400_%i' %fold, path_logdir='models/logs_kfold_mano_400_%i' %fold, epoch=30)
	test_model(model, X_test, batchSize, path_model='./models/model_kfold_mano_400_%i' %fold, outfile=('models/'+str+'_%i') %fold)
	#'''

'''
Epoch:  45 loss -  0.011424405111646562
Epoch:  46 loss -  0.011360000786273356
Epoch:  47 loss -  0.011289037974961644
Epoch:  48 loss -  0.011210053890355405
Epoch:  49 loss -  0.011121671371606166
Thresh -  0.005
94936   94936
Conf:  65529  --  129  --  35  --  29243
Thresh -  0.006
94936   94936
Conf:  67707  --  125  --  39  --  27065
Thresh -  0.007
94936   94936
Conf:  69487  --  116  --  48  --  25285
'''