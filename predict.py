# This Python file uses the following encoding: utf-8
from __future__ import division
import tensorflow as tf
import params
import numpy as np
import helpers
from levenhstein_distance import levenshtein_distance as l_distance
from pyvi.pyvi import ViTokenizer as tokenizer
from termcolor import colored
from dual_encoder import dual_encoder

embeddings_W = helpers.load_embeddings()
data = tf.placeholder(dtype=tf.int32, shape=[None, params.MAX_SEQ_LEN], name='data')
seq_len = tf.placeholder(dtype=tf.int32, shape=[None], name='seq_len')
targets = tf.placeholder(dtype=tf.int32, shape=[None,1], name='targets')

data_embed = tf.nn.embedding_lookup(embeddings_W, data, name="embed_context")
M = tf.get_variable("M",
                    shape=[params.RNN_DIM, params.RNN_DIM],
                    initializer=tf.truncated_normal_initializer())
'''
cell = tf.contrib.rnn.LSTMCell(
    params.RNN_DIM,
    forget_bias=2.0,
    use_peepholes=True,
    state_is_tuple=True)
     
outputs, states = tf.nn.dynamic_rnn(
    cell,
    data_embed,
    sequence_length=seq_len,
    dtype=tf.float32)
     
encoding_context, encoding_utterance = tf.split(states.h, 2, 0)

generated_response = tf.matmul(encoding_context, M)
generated_response = tf.expand_dims(generated_response, 2)
encoding_utterance = tf.expand_dims(encoding_utterance, 2)
     
logits = tf.matmul(generated_response, encoding_utterance, True)
logits = tf.squeeze(logits, [2])
'''
logits = dual_encoder(data_embed, seq_len, M)     
prob = tf.sigmoid(logits)
     
losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.to_float(targets), logits=logits)
     
mean_loss = tf.reduce_mean(losses)
optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(mean_loss)
  
init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    saver.restore(sess, './models/model10.ckpt')
    vocab = helpers.load_vocab()
    questions, responses = helpers.load_predefined_data()
    
    context = None
    stopwords = helpers.load_stop_words()
    while True:
    	context = raw_input("> ")
    	if context == "exit" or context == "EXIT":
    		break
    	context = tokenizer.tokenize(context.decode('utf-8'))
    	
    	distances = [l_distance(context.encode('utf-8'), q) for q in questions]
    	positions = np.array(distances).argsort()[-2:][::-1]
    	
    	if distances[positions[0]] >= params.top_threshold:
    		print colored(responses[positions[0]].replace('_', ' '), 'white')
    	elif distances[positions[0]] < params.top_threshold and distances[positions[0]] >= params.bottom_threshold:
    		max_prob = 0
    		idx = 0
    		context = helpers.remove_stop_words(sentence=context, stopwords=stopwords)
    		encoded_context = helpers.sentence2vector(context, vocab=vocab)
    		
    		for i in range(len(positions)):
    			batch_data = []
    			batch_data.extend(encoded_context)
    			encoded_response = helpers.sentence2vector(helpers.remove_stop_words(sentence=responses[positions[i]].decode('utf-8'), stopwords=stopwords), vocab=vocab)
    			sequence_len = [helpers.get_length_encoded_sentence(encoded_context), helpers.get_length_encoded_sentence(encoded_response)]
    			batch_data.extend(encoded_response)
    			probability = sess.run(prob, feed_dict={data: np.array(batch_data).reshape(2, params.MAX_SEQ_LEN), seq_len: sequence_len, targets: [[0]]})

    			if probability[0][0] > max_prob:
    				max_prob = probability
    				idx = positions[i]
    		print colored(responses[idx].replace('\n', ''), 'white')
    	else:
    		print colored(u"Tham khảo thêm trên trang hỗ trợ khách hàng của Samsung: https://samsung.com.vn hoặc liên lạc theo số 1900 1000", 'white')
    	del distances, positions
    	
