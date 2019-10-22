#! /usr/bin/env python3

import pickle
import numpy as np
import scipy.stats


def create_current(size):
	return np.eye(size) + 0.1

def create_previous(size):
	x = np.eye(size)
	x = np.roll(x, -1, axis=1)
	x[0, -1] = 0.0
	return x

def create_previous2(size):
	x = np.eye(size)
	x = np.roll(x, -2, axis=1)
	x[0, -2] = 0.0
	x[1, -1] = 0.0
	return x

def create_next(size):
	x = np.eye(size)
	x = np.roll(x, 1, axis=1)
	x[-1, 0] = 0.0
	return x

def create_next2(size):
	x = np.eye(size)
	x = np.roll(x, 2, axis=1)
	x[-2, 0] = 0.0
	x[-1, 1] = 0.0
	return x

def create_first(size):
	x = np.zeros((size, size))
	x[:,0] = 1.0
	return x

def create_firstFirstHalf(size):
	x = np.zeros((size, size))
	x[:size//2,0] = 1.0
	return x

def create_firstSecondHalf(size):
	x = np.zeros((size, size))
	x[(size+1)//2:,0] = 1.0
	return x

def create_last(size):
	x = np.zeros((size, size))
	x[:,-1] = 1.0
	return x

def create_lastFirstHalf(size):
	x = np.zeros((size, size))
	x[:size//2,-1] = 1.0
	return x

def create_lastSecondHalf(size):
	x = np.zeros((size, size))
	x[(size+1)//2:,-1] = 1.0
	return x

def create_random(size):
	x = np.random.rand(size, size)
	return x

# a = np.random.rand(5, 5)
# b = np.random.rand(5, 5)
# print(a, a.flatten())
# print(b, b.flatten())
# print(scipy.stats.pearsonr(a.flatten(), b.flatten()))

paths = ["pud.1ah.pkl", "pud.2ah.pkl", "pud.3ah.pkl", "pud.4ah.pkl", "pud.5ah.pkl", "pud.6ah.pkl", "pud.8ah.fullModel.pkl"]

for path in paths:
	print("File:", path)

	with open(path, 'rb') as f:
		sentenceList = pickle.load(f)
		nbHeads = sentenceList[0].shape[0]
		correlationTotals = {}
		for i in range(nbHeads):
			correlationTotals[i] = {"current": 0, "previous": 0, "previous2": 0, "next": 0, "next2": 0, "first": 0,
			"firstFirstHalf": 0, "firstSecondHalf": 0, "last": 0, "lastFirstHalf": 0, "lastSecondHalf": 0, "random": 0}
		for sentence in sentenceList:
			for headNr, attentionHead in enumerate(sentence):
				n = attentionHead.shape[0]
				for key in correlationTotals[headNr]:
					compMatrix = globals()["create_" + key](n)
					# Pearson's R only supports 1D vectors - just concatenate all rows together with flatten:
					correlation = scipy.stats.pearsonr(attentionHead.flatten(), compMatrix.flatten())[0]
					correlationTotals[headNr][key] += correlation
		
		for headNr in correlationTotals:
			for key in correlationTotals[headNr]:
				correlationTotals[headNr][key] /= len(sentenceList)
		
		for headNr in correlationTotals:
			maxKey = max(correlationTotals[headNr], key=correlationTotals[headNr].get)
			print("  Head {}: {} {:.2f}%".format(headNr+1, maxKey, 100*correlationTotals[headNr][maxKey]))
			#for key, value in sorted(correlationTotals[headNr].items(), key=lambda x: x[1], reverse=True):
			#	print("   ", key, correlationTotals[headNr][key])
		print()
