import tensorflow as tf
import os
import unicodedata
import re
import io
from sklearn.model_selection import train_test_split

# Download the file
path_to_zip = tf.keras.utils.get_file(
    'spa-eng.zip', origin='http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip',
    extract=True)

path_to_file = os.path.dirname(path_to_zip)+"/spa-eng/spa.txt"

# Converts the unicode file to ascii
def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')

def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)
    w = w.rstrip().strip()
    w = '<start> ' + w + ' <end>'
    return w


en_sentence = u"May I borrow this book?"
sp_sentence = u"¿Puedo tomar prestado este libro?"
print(preprocess_sentence(en_sentence))
print(preprocess_sentence(sp_sentence))


def create_dataset(path, nums=None):
    data = io.open(path, encoding='utf-8').read().strip().split('\n')
    clean_data = [[preprocess_sentence(s) for s in l.split('\t')] for l in data[:nums]]
    return zip(*clean_data)


en, sp = create_dataset(path_to_file, 1)
print(en[0])


def max_length(tensor):
    return max(len(t) for t in tensor)


def create_tokenizer(lang):
    tk = tf.keras.preprocessing.text.Tokenizer(filters='')
    tk.fit_on_texts(lang)
    tensor = tk.texts_to_sequences(lang)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,
                                                           padding='post')
    return tensor, tk

def load_dataset(path, nums):
    inp_lang, tar_lang = create_dataset(path, nums)
    inp_tensor, inp_tk = create_tokenizer(inp_lang)
    tar_tensor, tar_tk = create_tokenizer(tar_lang)
    return inp_tensor, tar_tensor, inp_tk, tar_tk


NUM_EXAMPLES = 30000
inp_tensor, tar_tensor, inp_tk, tar_tk = load_dataset(path_to_file, NUM_EXAMPLES)

max_len_inp = max_length(inp_tensor)
max_len_tar = max_length(tar_tensor)

inp_tensor_train, inp_tensor_val, tar_tensor_train, tar_tensor_val = train_test_split(inp_tensor, tar_tensor, test_size=0.2)
print(len(inp_tensor_train))

def convert(lang_tk, tensor):
    for i in tensor:
        if i != 0:
            print("%d --------------> %s" % (i, lang_tk.index_word[i]))

convert(inp_tk, inp_tensor_train[0])
convert(tar_tk, tar_tensor_train[0])


BUFFER_SIZE = len(inp_tensor_train)
BATCH_SIZE = 64
steps_per_epoch = BUFFER_SIZE // BATCH_SIZE
embedding_dim = 256
hidden_size = 1024
vocab_inp_size = len(inp_tk.index_word) + 1
vocab_tar_size = len(tar_tk.index_word) + 1

dataset = tf.data.Dataset.from_tensor_slices((inp_tensor_train, tar_tensor_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

example_input_batch, example_target_batch = next(iter(dataset))
# print(example_input_batch, example_target_batch)


class Encoder(tf.keras.Model):
    def __init__(self, batch_sz, emd_dim, hidden_sz, vocab_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.hidden_sz = hidden_sz
        self.embeddings = tf.keras.layers.Embedding(vocab_sz, emd_dim)
        self.gru = tf.keras.layers.GRU(self.hidden_sz,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

    def call(self, x, hidden):
        x = self.embeddings(x)
        output, state = self.gru(x, initial_state=hidden)
        return output, state

    def initial_hidden_state(self):
        return tf.zeros((self.batch_sz, self.hidden_sz))


encoder = Encoder(BATCH_SIZE, embedding_dim, hidden_size, vocab_inp_size)

sample_hidden = encoder.initial_hidden_state()
sample_output, sample_state = encoder(example_input_batch, sample_hidden)
print(sample_output.shape, sample_output)
print(sample_state.shape, sample_state)


class BahdanauAttention(tf.keras.Model):
    def __init__(self, hidden_sz):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(hidden_sz)
        self.W2 = tf.keras.layers.Dense(hidden_sz)
        self.V = tf.keras.layers.Dense(1)

    def call(self, x, hidden):
        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        hidden_with_time_axis = tf.expand_dims(hidden, 1)

        # score shape == (batch_size, max_len, hidden_size)
        score = self.V(tf.nn.tanh(self.W1(x) + self.W2(hidden_with_time_axis)))

        # attention_weights shape = (batch_size, max_len, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape = (batch_size, hidden_size)
        context_vector = attention_weights * x
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


attention_layer = BahdanauAttention(10)
attention_result, attention_weights = attention_layer(sample_output, sample_hidden)
print(attention_result.shape)
print(attention_weights.shape)


class Decoder(tf.keras.Model):
    def __init__(self, batch_sz, emd_dim, hidden_sz, vocab_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.hidden_sz = hidden_sz
        self.embeddings = tf.keras.layers.Embedding(vocab_sz, emd_dim)
        self.gru = tf.keras.layers.GRU(self.hidden_sz,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_sz)
        self.attention = BahdanauAttention(self.hidden_sz)

    def call(self, x, hidden, enc_output):
        # enc_output shape = (batch_sz, max_seq_len, hidden_sz)
        context_vector, attention_weights = self.attention(enc_output, hidden)

        # x shape after embeddings = (batch_sz, 1, embedding_dim)
        x = self.embeddings(x)

        # x shape after concat = (batch_sz, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        output, state = self.gru(x)

        # output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))

        # output shape = (batch_sz, vocab_sz)
        x = self.fc(output)

        return x, state, attention_weights


decoder = Decoder(BATCH_SIZE, embedding_dim, hidden_size, vocab_tar_size)
sample_decoder_output, _, _ = decoder(tf.random.uniform((64, 1)),
                                      sample_hidden, sample_output)
print(sample_decoder_output.shape)


# Define the optimizer and the loss function
optimizer = tf.kera

