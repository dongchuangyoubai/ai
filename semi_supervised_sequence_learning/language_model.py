import tensorflow as tf
from tensorflow.contrib import rnn

class LanguageModel(object):
    def __init__(self, word_dict, max_doc_len):
        self.embedding_size = 256
        self.hidden_size = 512

        self.x = tf.placeholder(tf.int32, [None, max_doc_len])

        with tf.name_scope('embeddings'):
            init_embeddings = tf.random_uniform([len(word_dict), self.embedding_size])
            embeddings = tf.get_variable('embeddings', initializer=init_embeddings)
            input_embeddings = tf.nn.embedding_look_up(embeddings, self.x)
