import tensorflow as tf
import pickle

class MatchLSTM(object):
    def __init__(self, dims, data, embeddings=0, q_seq_len=40, p_seq_len=653):
        self.q_seq_len = q_seq_len
        self.p_seq_len = p_seq_len
        self.batch_size = 1
        self.dims = dims
        self.embeddings = embeddings
        self.hidden_units = 128
        self.data = data.strip().split('|||')
        self.buildGraph()
        self.test()

    def buildGraph(self):
        self.setPlaceholder()
        self.getEmb()
        self.q_h = self.encoderQues()
        self.p_h = self.encoderPara()

    def setPlaceholder(self):
        self.para_inputs = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, None])
        self.ques_inputs = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, None])

    def getEmb(self):
        self.embeddings = tf.get_variable('embeddings', [201535, self.dims],
                                          initializer=tf.truncated_normal_initializer(stddev=0.1, seed=1), trainable=True)
        self.para_inputs_emds = tf.nn.embedding_lookup(self.embeddings, self.para_inputs)
        self.ques_inputs_embs = tf.nn.embedding_lookup(self.embeddings, self.ques_inputs)

    def encoderQues(self):
        with tf.variable_scope("Quesencoder"):
            cell = tf.contrib.rnn.LSTMCell(self.hidden_units)
            _, (c, h) = tf.nn.dynamic_rnn(
                cell,
                self.ques_inputs_embs,
                dtype=tf.float32
            )
            return h

    def encoderPara(self):
        with tf.variable_scope('Paraencoder'):
            cell = tf.contrib.rnn.LSTMCell(self.hidden_units)
            _, (c, h) = tf.nn.dynamic_rnn(
                cell,
                self.para_inputs_emds,
                dtype=tf.float32
            )
            return h




    def maskQues(self, ques):
        return ques + ['1' for _ in range(self.q_seq_len - len(ques))]

    def test(self):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            print(sess.run(self.q_h, feed_dict={self.ques_inputs: [self.data[1].split()], self.para_inputs: [self.data[0].split()]}))







if __name__ == '__main__':
    filename = 'demo_data'
    demo_data = open(filename, 'r', encoding='utf-8').readline()
    # [vocab_list, vocab_embs, word2id] = pickle.load(open('data.pickle', 'rb'))
    MatchLSTM(50, demo_data)