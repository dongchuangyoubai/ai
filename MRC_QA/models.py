import tensorflow as tf
import pickle
from data_utils import *
from tensorflow.contrib.seq2seq.python.ops.attention_wrapper import *
import numpy as np


class Encoder(object):
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size

    def encode(self, inputs, masks_inputs):
        ques_inputs, para_inputs = inputs
        ques_masks, para_masks = masks_inputs
        with tf.variable_scope("QuesEncoder"):
            cell = tf.contrib.rnn.LSTMCell(self.hidden_size)
            encoded_ques, (q_c, q_h) = tf.nn.dynamic_rnn(
                cell=cell,
                inputs=ques_inputs,
                sequence_length=ques_masks,
                dtype=tf.float32
            )
        with tf.variable_scope('ParaEncoder'):
            cell = tf.contrib.rnn.LSTMCell(self.hidden_size)
            encoded_para, (p_c, p_h) = tf.nn.dynamic_rnn(
                cell=cell,
                inputs=para_inputs,
                sequence_length=para_masks,
                dtype=tf.float32
            )
        return encoded_ques, encoded_para, q_c, p_c


class Decoder(object):
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size

    def decode(self, encoded, masks, labels):
        output_attender_fw = self.run_match_lstm(encoded, masks)
        logits = self.run_answer_ptr(output_attender_fw, masks, labels)
        return logits


    def run_match_lstm(self, encoded, masks):
        encoded_ques, encoded_para = encoded
        masks_question, masks_passage = masks
        with tf.variable_scope("match_lstm_attender"):
            attention = BahdanauAttention(
                self.hidden_size,
                encoded_ques,
                memory_sequence_length=masks_question
            )
            cell = tf.contrib.rnn.LSTMCell(self.hidden_size, state_is_tuple=True)
            decoder_cell = AttentionWrapper(
                cell,
                attention
            )
            output_attender_fw, _ = tf.nn.dynamic_rnn(decoder_cell, encoded_para, dtype=tf.float32, scope="rnn")
        return output_attender_fw

    def run_answer_ptr(self, inputs, masks, labels):
        masks_question, masks_passage = masks
        print(inputs.get_shape())
        print(labels.get_shape())
        # labels = tf.unstack(labels, axis=1)
        with tf.variable_scope("answer_ptr_attender"):
            attention = BahdanauAttention(
                self.hidden_size,
                inputs,
                memory_sequence_length=masks_question
            )
            cell = tf.contrib.rnn.LSTMCell(self.hidden_size, state_is_tuple=True)
            decoder_cell = AttentionWrapper(
                cell,
                attention
            )

            logits, _ = tf.nn.dynamic_rnn(decoder_cell, labels, dtype=tf.float32, scope="rnn")
        return logits






class MatchLSTM(object):
    def __init__(self, encoder, decoder, dims, data, embeddings=0, q_seq_len=40, p_seq_len=653):
        self.q_seq_len = q_seq_len
        self.p_seq_len = p_seq_len
        self.batch_size = 2
        self.dims = dims
        self.embeddings = embeddings
        self.hidden_size = 128
        self.encoder = encoder
        self.decoder = decoder
        self.data = data
        self.buildGraph()

    def buildGraph(self):
        self.setPlaceholder()
        self.getEmb()
        encoded_ques, encoded_para, q_c, p_c = self.encoder.encode(
            [self.ques_inputs_embs, self.para_inputs_emds], [self.ques_lengths, self.para_lengths])
        self.logits = self.decoder.decode([encoded_ques, encoded_para],
                                                      [self.ques_lengths, self.para_lengths], self.labels)
        # self.run_match_lstm([self.encoded_ques, self.encoded_para])

        self.test()

    def setPlaceholder(self):
        self.para_inputs = tf.placeholder(dtype=tf.int32, shape=[None, None])
        self.ques_inputs = tf.placeholder(dtype=tf.int32, shape=[None, None])
        self.ques_lengths = tf.placeholder(tf.int32, shape=[None], name="question_lengths")
        self.para_lengths = tf.placeholder(tf.int32, shape=[None], name="paragraph_lengths")
        self.labels = tf.placeholder(tf.int32, shape=[None, 2], name="gold_labels")
        self.dropout = tf.placeholder(tf.float32, shape=[], name="dropout")


    def getEmb(self):
        self.embeddings = tf.get_variable('embeddings', [300000, self.dims],
                                          initializer=tf.truncated_normal_initializer(stddev=0.1, seed=1), trainable=True)
        self.para_inputs_emds = tf.nn.embedding_lookup(self.embeddings, self.para_inputs)
        self.ques_inputs_embs = tf.nn.embedding_lookup(self.embeddings, self.ques_inputs)



    def getFeedDict(self, questions, paragraphs, answers, dropout_val):
        """
        -arg questions: A list of list of ids representing the question sentence
        -arg contexts: A list of list of ids representing the context paragraph
        -arg dropout_val: A float representing the keep probability for dropout

        :return: dict {placeholders: value}
        """

        padded_questions, question_lengths = pad_sequences(questions, 0)
        padded_contexts, passage_lengths = pad_sequences(paragraphs, 0)
        feed = {
            self.para_inputs: padded_questions,
            self.ques_inputs: padded_contexts,
            self.ques_lengths: question_lengths,
            self.para_lengths: passage_lengths,
            self.labels: answers,
            self.dropout: dropout_val
        }
        print(self.labels.get_shape())
        return feed

    def test(self):
        """
        valid: a list containing q, c and a.
        :return: loss on the valid dataset and the logit values
        """
        with tf.Session() as sess:
            p, q, a = self.data
            int_p = []
            int_q = []
            int_a = []
            print(a)
            p = [i.strip().split() for i in p]
            for i in p:
                int_p.append([int(j) for j in i])
            q = [i.strip().split() for i in q]
            for i in q:
                int_q.append([int(j) for j in i])
            a = [i.strip().split() for i in a]
            for i in a:
                int_a.append([int(j) for j in i])
            # at test time we do not perform dropout.
            print(int_a)
            sess.run(tf.global_variables_initializer())
            input_feed = self.getFeedDict(int_q, int_p, int_a, 1.0)
            print(sess.run(self.output_attender_fw, feed_dict=input_feed))
            # output_feed = [self.labels]

            # outputs = sess.run(output_feed, input_feed)

        # return outputs[0][0], outputs[0][1]









if __name__ == '__main__':
    filename = 'demo_data'
    demo_data = open(filename, 'r', encoding='utf-8').readlines()
    # [vocab_list, vocab_embs, word2id] = pickle.load(open('data.pickle', 'rb'))
    print(len(demo_data))
    p = [i.strip().split('|||')[0] for i in demo_data]
    q = [i.strip().split('|||')[1] for i in demo_data]
    a = [i.strip().split('|||')[2] for i in demo_data]
    # print(a)
    encoder = Encoder(128)
    decoder = Decoder(128)
    MatchLSTM(encoder, decoder, 50, [p, q, a])
