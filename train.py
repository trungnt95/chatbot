from __future__ import division
import tensorflow as tf
import params
import numpy as np
import helpers
from Data import DataObject
from dual_encoder import dual_encoder

data = tf.placeholder(dtype=tf.float32, shape=[None, params.MAX_SEQ_LEN, params.EMBEDDING_DIMEN], name='data')
seq_len = tf.placeholder(dtype=tf.int32, shape=[None], name='seq_len')
targets = tf.placeholder(dtype=tf.int32, shape=[None,1], name='targets')

M = tf.get_variable("M",
                    shape=[params.RNN_DIM, params.RNN_DIM],
                    initializer=tf.truncated_normal_initializer())
                    
logits = dual_encoder(data, seq_len, M)
prob = tf.sigmoid(logits)
     
losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.to_float(targets), logits=logits)
     
mean_loss = tf.reduce_mean(losses)
optimizer = tf.train.AdamOptimizer(learning_rate=params.LEARNING_RATE).minimize(mean_loss)
  
init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    trainset = DataObject()
    for i in range(params.nepochs):
        print("EPOCH {}".format(i+1))
        for j in range(8):
        	batch_data, sequence_len, labels = trainset.next(batch_size=params.BATCH_SIZE)
        	
        	_, loss = sess.run([optimizer, mean_loss], feed_dict={data: batch_data,seq_len: sequence_len, targets: labels})
        	
        	print("INFO: Step {} Loss: {}".format((j+1)*params.BATCH_SIZE, loss))
        savepath = saver.save(sess, './models/model{}.ckpt'.format(i+1))
        print("Save model at {}".format(savepath))
    

