# This Python file uses the following encoding: utf-8
import sys, os
import numpy as np
import params

def load_stop_words():
	with open('./data/stopwords.txt', 'r') as f:
		words = f.readlines()
	for i in range(len(words)):
		words[i] = words[i].decode('utf-8').replace('\n', '')
	return words
def remove_stop_words(sentence, stopwords):
	for w in stopwords:
		sentence = sentence.replace(w, '')
	return sentence
	
# Load word embeddings
def load_embeddings():
    embeddings_w = []
    with open(params.EMBEDDINGS_FILE, 'r') as f:
        lines = f.readlines()
        for line in lines:
            embeddings_w.append(list(map(float, line.split())))
    return np.array(embeddings_w, dtype='float32').reshape((len(embeddings_w), params.EMBEDDING_DIMEN))

# Load vocabulary to encode sentence
def load_vocab():
    with open(params.VOCAB_FILE, 'r') as f:
        lines = f.readlines()
    vocab = dict()
    reversed_vocab = dict()

    for idx, word in enumerate(lines):
        vocab[word.decode('utf-8').replace('\n', '')] = idx
        reversed_vocab[idx] = word.decode('utf-8').replace('\n', '')
    return vocab

# Transform sentence to vector
def sentence2vector(sentence, vocab):
    sentence = sentence.lower()
    sentence_len = len(sentence.split())
    encoded_sentence = []
    for word in sentence.split():
    	if len(encoded_sentence) >= params.MAX_SEQ_LEN:
    		break
        if word in vocab:
            encoded_sentence.append(vocab[word])
        else:
            encoded_sentence.append(vocab[u'_UNK'])
    # pad length sentence to 300
    while len(encoded_sentence)<params.MAX_SEQ_LEN:
        encoded_sentence.append(0)
        
    return encoded_sentence

# Embed sentence with embeddings matrix
def embed_sentence(encoded_sentence, embeddings_w):
    for i in range(len(encoded_sentence)):
        encoded_sentence[i] = embeddings_w[encoded_sentence[i]]
    return encoded_sentence

# Convert list of String to list of int
def stringlist2intlist(data):
    return map(int, data)

def load_data(filename=params.TRAIN_FILE):
    with open(filename, 'r') as f:
        data = f.readlines()
    for i in range(len(data)):
        data[i] = data[i].replace('\n', '')

    data = [line.split(',') for line in data]
    for i in range(len(data)):
        question = map(int, data[i][0].split())
        answer = map(int, data[i][1].split())
        label = map(int, data[i][2])
        data[i] = (question, answer, label[0])
    return data

def get_length_encoded_sentence(sentence):
	if 0 in sentence:
		return sentence.index(0)
	return len(sentence)
# Load only data of responses
def load_predefined_data(filename=params.DATA_FILE):
	with open(filename, 'r') as f:
		data = f.readlines()
	data = [data[i].replace('\n', '') for i in range(len(data))]
	return [data[i] for i in range(len(data)) if i%2==0], [data[i] for i in range(len(data)) if i%2!=0]
