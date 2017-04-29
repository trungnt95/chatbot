import tensorflow as tf
import params
def dual_encoder(_data, _seq_len, _weights):
	cell = tf.contrib.rnn.LSTMCell(
		params.RNN_DIM,
		forget_bias=2.0,
		use_peepholes=True,
		state_is_tuple=True)
		 
	outputs, states = tf.nn.dynamic_rnn(
		cell,
		_data,
		sequence_length=_seq_len,
		dtype=tf.float32)
		 
	encoding_context, encoding_utterance = tf.split(states.h, 2, 0)
		 
	generated_response = tf.matmul(encoding_context, _weights)
	generated_response = tf.expand_dims(generated_response, 2)
	encoding_utterance = tf.expand_dims(encoding_utterance, 2)
		 
	logits = tf.matmul(generated_response, encoding_utterance, True)
	logits = tf.squeeze(logits, [2])
	return logits
