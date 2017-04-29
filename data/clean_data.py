# This Python file uses the following encoding: utf-8
#from __future__ import print_function
#from gensim.models import Word2Vec
import re

stopwords = []
with open('stopwords.txt', 'r') as f:
	stopwords = f.readlines()
for i in range(len(stopwords)):
	stopwords[i] = stopwords[i].decode('utf-8').replace('\n', '')

sentences = []
with open('neg.txt', 'r') as f:
	sentences = f.readlines()
for i in range(len(sentences)):
	sentences[i] = sentences[i].decode('utf-8').lower().replace('\n', '')

def clean_sentence(raw_sentence):
	global stopwords
	# remove non-letters
	letter_only = re.sub('[{}\[\]()-,\<>":.~!@#$%^&*+/?;0123456789]', " ", raw_sentence)
	words = letter_only.split()
	meaningful_words = [w for w in words if not w in stopwords]
	return " ".join(meaningful_words)

sentences = [clean_sentence(s) for s in sentences]
with open('neg.out', 'w') as f:
	for s in sentences:
		f.write(s.encode('utf-8'))
		f.write('\n')
