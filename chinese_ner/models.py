import tensorflow as tf

class NERModel(object):
    def __init__(self, batch_size, sequence_len):
        self.batch_size = batch_size
        self.sequence_len = sequence_len
        self.set_placeholder()

    def set_placeholder(self):
        self.inputs = tf.placeholder(tf.int32, shape=[self.batch_size, self.sequence_len])
        self.outputs = tf.placeholder(tf.int32, shape=[self.batch_size, self.sequence_len])