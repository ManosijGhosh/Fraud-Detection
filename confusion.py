import sys
import os
import numpy as np

def read_res(filename):
	i=0
	costs = np.array([],  dtype=np.float32)
	with open(filename, 'r+') as file:
		for line in file:
			i+=1
			if i%2 == 0:
				temp = str(line[line.find("[")+1:line.find("]")])
				arr = np.fromstring(temp, dtype=np.float32, sep = ', ')
				costs = np.append(costs, arr)
				#print(arr)
	'''			
	print(costs)
	print(len(costs))
	print(len(costs)/32)	
	print(i)
	'''
	return costs

def evaluate(file, thresh = 3):

	labels = list()

	costs = read_res(file)
	for i in range(len(costs)):
		
		if costs[i]>thresh:
			labels.append(1)
		else:
			labels.append(0)

	return labels
#main()