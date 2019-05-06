import tensorflow as tf

class NERModel(object):
    def __init__(self, batch_size, sequence_len, vocab_size, embedding_dim=5):
        self.batch_size = batch_size
        self.sequence_len = sequence_len
        self.embedding_size = vocab_size
        self.embedding_dim = embedding_dim
        self.set_placeholder()
        self.test()

    def set_placeholder(self):
        embeddings = tf.get_variable('embeddings', [self.embedding_size, self.embedding_dim],
                                          initializer=tf.truncated_normal_initializer(stddev=0.1, seed=1))
        self.inputs = tf.placeholder(tf.int32, shape=[self.batch_size, self.sequence_len])
        self.inputs_embeddings = tf.nn.embedding_lookup(embeddings, self.inputs)
        self.outputs = tf.placeholder(tf.int32, shape=[self.batch_size, self.sequence_len])

    def test(self):
        test_x = [[1, 2, 3], [4, 5, 6]]
        for i in test_x:
            if len(i) < self.sequence_len:
                i.extend([0 for _ in range(self.sequence_len - len(test_x) - 1)])
        print(test_x)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            print(sess.run(self.inputs_embeddings, feed_dict={self.inputs: test_x}))





if __name__ == '__main__':
    ner = NERModel(2, 10, 10)