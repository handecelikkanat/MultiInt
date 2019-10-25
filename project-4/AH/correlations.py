#! /usr/bin/env python3

import pickle
import numpy as np
import scipy.stats


def create_current(size, tokens):
	return np.eye(size) + 0.1

def create_previous(size, tokens):
	x = np.eye(size)
	x = np.roll(x, -1, axis=1)
	x[0, -1] = 0.0
	return x

def create_previous2(size, tokens):
	x = np.eye(size)
	x = np.roll(x, -2, axis=1)
	x[0, -2] = 0.0
	x[1, -1] = 0.0
	return x

def create_next(size, tokens):
	x = np.eye(size)
	x = np.roll(x, 1, axis=1)
	x[-1, 0] = 0.0
	return x

def create_next2(size, tokens):
	x = np.eye(size)
	x = np.roll(x, 2, axis=1)
	x[-2, 0] = 0.0
	x[-1, 1] = 0.0
	return x

def create_first(size, tokens):
	x = np.zeros((size, size))
	x[:,0] = 1.0
	return x

def create_firstFirstHalf(size, tokens):
	x = np.zeros((size, size))
	x[:size//2,0] = 1.0
	return x

def create_firstSecondHalf(size, tokens):
	x = np.zeros((size, size))
	x[(size+1)//2:,0] = 1.0
	return x

def create_last(size, tokens):
	x = np.zeros((size, size))
	x[:,-1] = 1.0
	return x

def create_lastFirstHalf(size, tokens):
	x = np.zeros((size, size))
	x[:size//2,-1] = 1.0
	return x

def create_lastSecondHalf(size, tokens):
	x = np.zeros((size, size))
	x[(size+1)//2:,-1] = 1.0
	return x

def create_random(size, tokens):
	x = np.random.rand(size, size)
	return x

def create_samewordbpe(size, tokens):
	x = np.zeros((size, size))
	for i in range(size):
		if tokens[i].endswith("@@"):
			x[i,i+1] = 1.0
			x[i+1,i] = 1.0
			x[i,i] = 1.0
			x[i+1, i+1] = 1.0
			if tokens[i+1].endswith("@@"):
				x[i,i+2] = 1.0
				x[i+2,i] = 1.0
	return x

# a = np.random.rand(5, 5)
# b = np.random.rand(5, 5)
# print(a, a.flatten())
# print(b, b.flatten())
# print(scipy.stats.pearsonr(a.flatten(), b.flatten()))


#paths = ["pud.1ah.pkl", "pud.2ah.pkl", "pud.3ah.pkl", "pud.4ah.pkl", "pud.5ah.pkl", "pud.6ah.pkl", "pud.8ah.fullModel.pkl"]

bpepath = "en_pud-ud-test.conllu.tok.tc.bpe"


def loadSentences(bpepath):
	sentences = []
	with open(bpepath, 'r') as f:
		for line in f:
			tokens = line.strip().split(" ")
			sentences.append(tuple(tokens))
	return sentences


def analyzeMatrix(path, sentences):
	print("File:", path)

	with open(path, 'rb') as f:
		attentionList = pickle.load(f)
		nbHeads = attentionList[0].shape[0]
		correlationTotals = {}
		for i in range(nbHeads):
			correlationTotals[i] = {"current": 0, "previous": 0, "previous2": 0, "next": 0, "next2": 0, "first": 0,
			"firstFirstHalf": 0, "firstSecondHalf": 0, "last": 0, "lastFirstHalf": 0, "lastSecondHalf": 0, "random": 0, "samewordbpe": 0}
		for attentionMatrix, sentence in zip(attentionList, sentences):
			for headNr, attentionHead in enumerate(attentionMatrix):
				n = attentionHead.shape[0]
				for key in correlationTotals[headNr]:
					compMatrix = globals()["create_" + key](n, sentence)
					# Pearson's R only supports 1D vectors - just concatenate all rows together with flatten:
					correlation = scipy.stats.pearsonr(attentionHead.flatten(), compMatrix.flatten())[0]
					correlationTotals[headNr][key] += correlation
		
		for headNr in correlationTotals:
			for key in correlationTotals[headNr]:
				correlationTotals[headNr][key] /= len(attentionList)
		
		for headNr in correlationTotals:
			#maxKey = max(correlationTotals[headNr], key=correlationTotals[headNr].get)
			#print("  Head {}: {} {:.2f}%".format(headNr+1, maxKey, 100*correlationTotals[headNr][maxKey]))
			valueList = []
			for key, value in sorted(correlationTotals[headNr].items(), key=lambda x: x[1], reverse=True):
				if value > 0.3:
					valueList.append("{} {:.2f}%".format(key, 100*value))
			print("  Head {}:".format(headNr+1), " / ".join(valueList))
		print()


def analyze1Layer():
	paths = ["pud.1ah.pkl", "pud.2ah.pkl", "pud.3ah.pkl", "pud.4ah.pkl", "pud.5ah.pkl", "pud.6ah.pkl", "pud.8ah.fullModel.pkl"]
	sentences = loadSentences("en_pud-ud-test.conllu.tok.tc.bpe")
	for path in paths:
		analyzeMatrix(path, sentences)
	

def analyze6Layers():
	paths = ["6l/pud.0l.8ah.fullModel.pkl", "6l/pud.1l.8ah.fullModel.pkl", "6l/pud.2l.8ah.fullModel.pkl", "6l/pud.3l.8ah.fullModel.pkl", "6l/pud.4l.8ah.fullModel.pkl", "6l/pud.5l.8ah.fullModel.pkl"]
	sentences = loadSentences("en_pud-ud-test.conllu.tok.tc.bpe")
	for path in paths:
		analyzeMatrix(path, sentences)


if __name__ == "__main__":
	#analyze1Layer()
	analyze6Layers()
	