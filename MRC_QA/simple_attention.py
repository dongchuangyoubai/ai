import tensorflow as tf
from tensorflow.contrib.seq2seq.python.ops.attention_wrapper import *

filename = 'demo_data'
demo_data = open(filename, 'r', encoding='utf-8').readlines()
# [vocab_list, vocab_embs, word2id] = pickle.load(open('data.pickle', 'rb'))
# print(len(demo_data))
p = [i.strip().split('|||')[0] for i in demo_data]
q = [i.strip().split('|||')[1] for i in demo_data]
a = [i.strip().split('|||')[2] for i in demo_data]

print(len(p[0]))
int_p = []
p = [i.strip().split() for i in p]
for i in p:
    int_p.append([int(j) for j in i])
print(int_p)

batch_size = len(demo_data)
encoder_input = tf.placeholder(tf.int32, [batch_size, None])
decoder_target = tf.placeholder(tf.int32, [batch_size, None])

input_vocab_size = 400000
input_embedding_size = 128


embeddings = tf.Variable(tf.random_uniform(shape=[input_vocab_size, input_embedding_size],
                                           minval=-1.0, maxval=1.0), dtype=tf.float32)

input_embedd = tf.nn.embedding_lookup(embeddings, encoder_input)
target_embedd = tf.nn.embedding_lookup(embeddings, decoder_target)

hidden_size = 128
encoder_cell = tf.contrib.rnn.LSTMCell(hidden_size)
init_state = encoder_cell.zero_state(batch_size, dtype=tf.float32)
encoder_output, encoder_state = tf.nn.dynamic_rnn(encoder_cell, input_embedd, initial_state=init_state)

seq_len = tf.constant([525, 525], dtype=tf.int32)

d_cell = tf.contrib.rnn.LSTMCell(hidden_size)

attention_mechanism = BahdanauAttention(hidden_size, encoder_output,  memory_sequence_length=seq_len)
att_wrapper = AttentionWrapper(d_cell, attention_mechanism, cell_input_fn=lambda input, attention: input)
states = att_wrapper.zero_state(batch_size, tf.float32)
# output_attender, _ = tf.nn.dynamic_rnn(att_wrapper, encoder_output, dtype=tf.float32)
for i in range(batch_size):
    h_bar_without_tanh, states = att_wrapper(encoder_output[i], states)
    h_bar = tf.tanh(h_bar_without_tanh)
    _X = tf.nn.softmax(tf.matmul(h_bar, W), 1)
    print(h_bar_without_tanh)
    print(h_bar)