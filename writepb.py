import tensorflow as tf
from tensorflow.python.framework import graph_util

filedir = "./models/model10.ckpt"
output_graph = "./models/frozen_model.pb"
saver = tf.train.import_meta_graph("./models/model10.ckpt.meta", clear_devices=True)

graph = tf.get_default_graph()
input_graph_def = graph.as_graph_def()

with tf.Session() as sess:
	saver.restore(sess, filedir)
	for x in tf.global_variables():
		print x
	output_graph_def = graph_util.convert_variables_to_constants(
		sess,
		input_graph_def,
		["M","rnn/lstm_cell/w_f_diag"])
	test = tf.get_default_graph().get_tensor_by_name("rnn/lstm_cell/biases/read:0").eval()
	print test
		
	with tf.gfile.GFile(output_graph, "wb") as f:
		f.write(output_graph_def.SerializeToString())
		


