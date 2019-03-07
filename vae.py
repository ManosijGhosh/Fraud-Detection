import numpy as np
import signal
import sys
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from confusion import evaluate

def make_data():
	RANDOM_SEED = 42
	df = pd.read_csv('creditcard.csv')
	data = df
	data['Time'] = data['Time'].apply(lambda x : ((x//60)%1440)/1440) # convert to time of day in minuites
	data['Amount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))
	# Prepare for train
	X_train, X_test = train_test_split(data, test_size=0.3, random_state=RANDOM_SEED)
	X_train = X_train[X_train.Class == 0]
	X_train = X_train.drop(['Class'], axis=1)
	Y_test = X_test['Class']
	X_test = X_test.drop(['Class'], axis=1)
	X_train = X_train.values
	X_test = X_test.values
	Y_test = Y_test.values

	return X_test, X_train, Y_test

def create_model(X_test, X_train, Y_test, lrate=1e-5):
	model = dict()

	dim_b = 32
	dim_input = X_train.shape[1]
	dim_enc_1 = 17
	dim_enc_2 = 7
	dim_dec_1 = 17
	dim_dec_2 = dim_input

	#create model
	with tf.name_scope('input'):
		model['ip'] = tf.placeholder(tf.float32, [dim_b, dim_input], name = 'input-vector')

	# FIRST ENCODING LAYER

	with tf.name_scope('encoding-layer-1'):
		model['We_1'] = tf.Variable(tf.random_normal([dim_input, dim_enc_1], stddev=1.0/dim_enc_1), name = 'We-1')
		model['Be_1'] = tf.Variable(tf.random_normal([1, dim_enc_1], stddev=1.0/dim_enc_1), name = 'Be-1')
		model['ye_1'] = tf.nn.tanh(tf.add(tf.matmul(model['ip'], model['We_1']), model['Be_1']), name = 'ye-1')

	# SECOND ENCODING LAYER

	with tf.name_scope('encoding-layer-2'):
		model['We_2'] = tf.Variable(tf.random_normal([dim_enc_1, dim_enc_2], stddev=1.0/dim_enc_2), name = 'We-2')
		model['Be_2'] = tf.Variable(tf.random_normal([1, dim_enc_2], stddev=1.0/dim_enc_2), name = 'Be-2')
		#model['ye_2'] = tf.nn.tanh(tf.add(tf.matmul(model['ye_1'], model['We_2']), model['Be_2']), name = 'ye-2')
		model['mean-hidden'] = tf.nn.tanh(tf.add(tf.matmul(model['ye_1'], model['We_2']), model['Be_2']), name = 'mean')
		model['stddev-hidden'] = tf.nn.tanh(tf.add(tf.matmul(model['ye_1'], model['We_2']), model['Be_2']), name = 'stddev')

	# FIRST DECODING LAYER

	

	with tf.name_scope('sample-normal'):

		model['samples'] = tf.random_normal([dim_b, dim_enc_2], 0, 1, dtype=tf.float32)
		model['guess'] = tf.add(tf.multiply(model['stddev-hidden'], model['samples']), model['mean-hidden'], name='guess')

	## WARNING, POORLY WRITTEN CODE AHEAD. BLAME ME
	## REMOVE THIS FOR TRAINING.
	## ADD THIS LINE FOR TESTING
	#model['guess'] = model['mean-hidden']

	with tf.name_scope('decoding-layer-1'):
		model['Wd_1'] = tf.Variable(tf.random_normal([dim_enc_2, dim_dec_1], stddev=1.0/dim_dec_1), name = 'Wd-1')
		model['Bd_1'] = tf.Variable(tf.random_normal([1, dim_dec_1], stddev=1.0/dim_dec_1), name = 'Bd-1')
		model['yd_1'] = tf.nn.tanh(tf.add(tf.matmul(model['guess'], model['Wd_1']), model['Bd_1']), name = 'yd-1')
	
	# SECOND DECODING LAYER

	with tf.name_scope('decoding-layer-2'):
		model['Wd_2'] = tf.Variable(tf.random_normal([dim_dec_1, dim_dec_2], stddev=1.0/dim_dec_2), name = 'Wd-2')
		model['Bd_2'] = tf.Variable(tf.random_normal([1, dim_dec_2], stddev=1.0/dim_dec_2), name = 'Bd-2')

	with tf.name_scope('output'):
		model['op'] = tf.nn.tanh(tf.add(tf.matmul(model['yd_1'], model['Wd_2']), model['Bd_2']), name = 'output-vector')
		#model['op'] = tf.Print(model['op'], [model['op']], message="This is op: ", summarize=960)


	with tf.name_scope('loss_optim_4'):

		#model['cost'] = tf.reduce_mean(tf.pow(model['ip'] - model['op'], 2), name='cost')
		#model['regen-cost'] = tf.reduce_sum(tf.squared_difference(model['ip'], model['op']), name = 'reconstruction_cost')

		model['regen-cost'] = -tf.reduce_sum(model['ip'] * tf.log(1e-6 + model['op']) + (1-model['ip']) * tf.log(1e-6 + 1 - model['op']),1)

		#model['latent-cost'] = -0.5 * tf.reduce_sum(1.0 + 2.0 * model['stddev-hidden'] - tf.square(model['mean-hidden']) - tf.exp(2.0 * model['stddev-hidden']), 1)

		model['latent-cost'] = 0.5 * tf.reduce_sum(tf.square(model['mean-hidden']) + tf.square(model['stddev-hidden']) - tf.log(tf.square(model['stddev-hidden'])) - 1,1)

		model['total-cost'] = tf.reduce_mean(tf.add(model['regen-cost'], model['latent-cost']), name='total_cost')

		#model['cost-2'] = tf.reduce_mean(tf.squared_difference(model['ip'], model['op']), axis=1, name='cost-2')
		model['cost-2'] = -tf.reduce_mean(tf.reduce_sum(model['ip'] * tf.log(1e-8 + model['op']) + (1-model['ip']) * tf.log(1e-8 + 1 - model['op']),1),axis = 1, name='cost_2')
		model['optimizer'] = tf.train.AdamOptimizer(lrate).minimize(model['total-cost'], name = 'optim')
		model['sum_loss'] = tf.summary.scalar(model['total-cost'].name, model['total-cost'])
		model['print-cost'] = tf.Print(model['cost-2'], [model['cost-2']], message="This is cost: ", summarize = 32)

	# return the model dictionary

	return model;
def train_model(model, X_train, epoch = 50):

	with tf.Session() as session:
		init = tf.global_variables_initializer()
		session.run(init)

		path_model = './model-vae-5'
		path_logdir = 'logs-auto-vae-5'

		saver = tf.train.Saver()
		writer = tf.summary.FileWriter(path_logdir, session.graph)

		for i in range(epoch):
			for count in range(0,X_train.shape[0] - 31,32):
				in_vector = X_train[count:count+32]
				#in_vector = in_vector.reshape(1, in_vector.shape[0])
				feed = {model['ip']: in_vector}

				_, summary = session.run([model['optimizer'], model['sum_loss']], feed_dict = feed)
				#print('Cost: ', model['cost'].eval())
			writer.add_summary(summary, i)

			print('Epoch: ', i)

		saver.save(session, path_model)

		print('Parameters: ')
		print('We 1 \n ------------------------')
		print(model['We_1'].eval())
		print('Be 1 \n ------------------------')
		print(model['Be_1'].eval())
		print('We 2 \n ------------------------')
		print(model['We_2'].eval())
		print('Be 2 \n ------------------------')
		print(model['Be_2'].eval())
		print('Wd 1 \n ------------------------')
		print(model['Wd_1'].eval())
		print('Bd 1 \n ------------------------')
		print(model['Bd_1'].eval())
		print('Wd 2 \n ------------------------')
		print(model['Wd_2'].eval())
		print('Bd 2 \n ------------------------')
		print(model['Bd_2'].eval())

def test_model(model, X_test, Y_test):

	path_model = './model-vae-5'
	#path_logdir = 'logs-auto-2'

	with open('logfile-vae-5.txt','w+') as outfile:
		with tf.Session() as sess:

			saver = tf.train.Saver()
			saver.restore(sess, path_model)

			for i in range(0,X_test.shape[0]-2,32):

				in_vector = X_test[i:i+32]
				feed = {model['ip']: in_vector}

				outfile.write('Batch --- ' + str(i//32) +' --- \n')
				ans = sess.run(model['cost-2'], feed_dict =  feed)
				#print(list(ans))
				outfile.write(str(list(ans)))
				outfile.write('\n')

def consfusion_eval(labels, file):

	#sprint(labels)
	TPX, TNX, FPX, FNX = list(), list(), list(), list()
	for i in range(0,100):
		tp, tn, fp, fn = evaluate(labels, file, i/10.0)
		print('Thresh: ', i/10, 'Conf: ', tp, ' -- ', tn, ' -- ', fp, ' -- ', fn)
		TPX.append(tp)
		TNX.append(tn)
		FPX.append(fp)
		FNX.append(fn)

	return (TPX, TNX, FPX, FNX)

def main():
	
	X_test, X_train, Y_test = make_data()

	model = create_model(X_test, X_train, Y_test)
	train_model(model, X_train, epoch = 20)
	test_model(model, X_test, Y_test)
	consfusion_eval(Y_test, 'logfile-vae-5.txt')

main()
