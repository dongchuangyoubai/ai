import tensorflow as tf
from tensorflow.keras.layers import LSTMCell
from tensorflow.contrib import rnn


class AutoEncoder(object):
    def __init__(self, word_dict, max_doc_len):
        self.embedding_size = 256
        self.hidden_size = 512
        self.voc_size = len(word_dict)
        self.max_doc_len = max_doc_len
        self.word_dict = word_dict

        self.x = tf.placeholder(tf.int32, [None, max_doc_len])
        self.batch_size = tf.shape(self.x)[0]

        self.decoder_input = tf.concat([tf.ones([self.batch_size, 1], tf.int32) * self.word_dict['<s>'], self.x], axis=1)
        self.targets = tf.concat([self.x, tf.ones([self.batch_size, 1], tf.int32) * self.word_dict['</s>']], axis=1)

        self.encoder_input_len = tf.reduce_sum(tf.sign(self.x), 1)
        self.decoder_input_len = tf.reduce_sum(tf.sign(self.decoder_input), 1)

        with tf.variable_scope("embedding"):
            init_embeddings = tf.random_uniform([self.voc_size, self.embedding_size])
            embeddings = tf.get_variable("embeddings", initializer=init_embeddings)
            encoder_input_emd = tf.nn.embedding_lookup(embeddings, self.x)
            decoder_input_emd = tf.nn.embedding_lookup(embeddings, self.decoder_input)

        with tf.variable_scope("encoder"):
            encoder_cell = rnn.BasicLSTMCell(self.hidden_size)
            _, encoder_hidden_states = tf.nn.dynamic_rnn(
                encoder_cell, encoder_input_emd, sequence_length=self.encoder_input_len, dtype=tf.float32
            )

        with tf.variable_scope("decoder"):
            decoder_cell = rnn.BasicLSTMCell(self.hidden_size)
            decoder_output, _ = tf.nn.dynamic_rnn(
                decoder_cell, decoder_input_emd, sequence_length=self.decoder_input_len,
                initial_state=encoder_hidden_states, dtype=tf.float32
            )

        with tf.name_scope("output"):
            self.logits = tf.layers.dense(decoder_output, self.voc_size)


        with tf.name_scope("loss"):
            print(self.logits.shape)
            print(self.targets.shape)
            losses = tf.contrib.seq2seq.sequence_loss(
                logits=self.logits,
                targets=self.targets,
                weights=tf.sequence_mask(self.decoder_input_len, self.max_doc_len + 1, dtype=tf.float32),
                average_across_timesteps=False,
                average_across_batch=True
            )
            self.loss = tf.reduce_mean(losses)


