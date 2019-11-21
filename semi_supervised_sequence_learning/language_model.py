import tensorflow as tf
from tensorflow.contrib import rnn

class LanguageModel(object):
    def __init__(self, word_dict, max_doc_len):
        self.embedding_size = 256
        self.hidden_size = 512

        self.x = tf.placeholder(tf.int32, [None, max_doc_len])
        self.batch_size = tf.shape(self.x)[0]
        self.vocab_size = len(word_dict)

        self.inputs = tf.concat([self.x, tf.ones([self.batch_size, 1], dtype=tf.int32) * word_dict['</s>']], axis=1)
        self.targets = tf.concat([tf.ones([self.batch_size, 1], dtype=tf.int32) * word_dict['<s>'], self.x], axis=1)
        self.seq_len = tf.reduce_sum(tf.sign(self.inputs), 1)

        with tf.name_scope('embeddings'):
            init_embeddings = tf.random_uniform([len(word_dict), self.embedding_size])
            embeddings = tf.get_variable('embeddings', initializer=init_embeddings)
            input_embeddings = tf.nn.embedding_lookup(embeddings, self.inputs)

        with tf.name_scope('rnn'):
            cell = rnn.BasicLSTMCell(self.hidden_size)
            outputs, _ = tf.nn.dynamic_rnn(
                cell, input_embeddings, sequence_length=self.seq_len, dtype=tf.float32
            )

        with tf.name_scope('outputs'):
            self.logits = tf.layers.dense(outputs, self.vocab_size)

        with tf.name_scope('loss'):
            losses = tf.contrib.seq2seq.sequence_loss(
                logits=self.logits,
                targets=self.targets,
                weights=tf.sequence_mask(self.seq_len, max_doc_len + 1, dtype=tf.float32),
                average_across_timesteps=False,
                average_across_batch=True
            )
            self.loss = tf.reduce_mean(losses)
