# This Python file uses the following encoding: utf-8
import helpers
import numpy as np
import params

class DataObject(object):
	
	def __init__(self):
		self.batch_idx = 0
		self.questions = []
		self.responses = []
		self.labels = []
		self.embeddings = helpers.load_embeddings()
		data = helpers.load_data()
		for item in data:
			self.questions.append(item[0])
			self.responses.append(item[1])
			self.labels.append(item[2]) 
		del data
	
	def next(self, batch_size):
		if self.batch_idx == len(self.questions):
			self.batch_idx = 0
		
		q = self.questions[self.batch_idx:min(self.batch_idx + batch_size, len(self.questions))]
		r = self.responses[self.batch_idx:min(self.batch_idx + batch_size, len(self.questions))]
		batch_data = []
		for v in q:
			for idx in v:
				batch_data.extend(self.embeddings[idx])
		for v in r:
			for idx in v:
				batch_data.extend(self.embeddings[idx])
				
		batch_data = np.array(batch_data).reshape(2 * batch_size, params.MAX_SEQ_LEN, params.EMBEDDING_DIMEN)
		sequence_len = []
		for item in q:
			sequence_len.append(helpers.get_length_encoded_sentence(list(item)))
		for item in r:
			sequence_len.append(helpers.get_length_encoded_sentence(list(item)))
		batch_labels = self.labels[self.batch_idx:min(self.batch_idx + batch_size, len(self.questions))]
		self.batch_idx = min(self.batch_idx + batch_size, len(self.questions))
		return batch_data, sequence_len, np.array(batch_labels).reshape(batch_size, 1)
	
