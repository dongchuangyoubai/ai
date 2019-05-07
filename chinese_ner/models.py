import tensorflow as tf

class NERModel(object):
    def __init__(self, batch_size, sequence_len, vocab_size, embedding_dim=50):
        self.batch_size = batch_size
        self.sequence_len = sequence_len
        self.embedding_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_units = 128
        self.dropout = 0.8
        self.tags = 10
        self.set_placeholder()
        self.build_net()

    def set_placeholder(self):
        embeddings = tf.get_variable('embeddings', [self.embedding_size, self.embedding_dim],
                                          initializer=tf.truncated_normal_initializer(stddev=0.1, seed=1), trainable=True)
        self.inputs = tf.placeholder(tf.int32, shape=[self.batch_size, self.sequence_len])
        self.inputs_embeddings = tf.nn.embedding_lookup(embeddings, self.inputs)
        self.labels = tf.placeholder(tf.int64, shape=[self.batch_size, self.sequence_len])

    def build_net(self):
        cell = tf.contrib.rnn.LSTMCell(self.hidden_units)
        output_lstm, _ = tf.nn.dynamic_rnn(
            cell,
            self.inputs_embeddings,
            [self.sequence_len for _ in range(self.batch_size)],
            dtype=tf.float32
            )
        output_dropout = tf.nn.dropout(output_lstm, self.dropout)

        w = tf.get_variable(name='w',
                            shape=[self.hidden_units, self.tags],
                            initializer=tf.contrib.layers.xavier_initializer(),
                            dtype=tf.float32)
        b = tf.get_variable(name='b',
                            shape=[self.tags],
                            initializer=tf.zeros_initializer(),
                            dtype=tf.float32)
        # logits = [tf.matmul(item, w) + b for item in output_dropout]
        logits = tf.map_fn(lambda item: tf.matmul(item, w) + b, output_dropout)
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=logits)
        loss = tf.reduce_mean(losses)

        if self.optimizer == 'Adam':
            optimizer = tf.train.AdamOptimizer(self.decay_lr)
        elif self.optimizer == 'Adadelta':
            optimizer = tf.train.AdadeltaOptimizer(self.decay_lr)
        train_op = optimizer.minimize(loss)

        pred = tf.argmax(logits, axis=-1)
        accuracy = tf.equal(pred, self.labels)
        accuracy = tf.reduce_mean(tf.cast(accuracy, tf.float32))

        fr = open('train_data_x', 'r', encoding='utf-8')
        train_x = []
        for i in fr.readlines():
            train_x.append([int(j) for j in i.strip().split()])
        fr.close()
        fr = open('train_data_y', 'r', encoding='utf-8')
        train_y = []
        for i in fr.readlines():
            train_y.append([int(j) for j in i.strip().split()])

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(10):
                for i in range(int(46364/self.batch_size)):
                    x = train_x[i: i + self.batch_size]
                    y = train_y[i: i + self.batch_size]
                    for j in x:
                        if len(j) < self.sequence_len:
                            j.extend([0 for _ in range(self.sequence_len - len(j))])
                    for j in y:
                        if len(j) < self.sequence_len:
                            j.extend([0 for _ in range(self.sequence_len - len(j))])
                    sess.run(train_op, feed_dict={self.inputs: x, self.labels: y})
                    if i % 100 == 0:
                        print(sess.run((loss, accuracy), feed_dict={self.inputs: x, self.labels: y}))
                break



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
    ner = NERModel(32, 581, 4000)