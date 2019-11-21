import math
import torch
import torch.nn as nn
from loss_func import NegativeSampling

class Model(nn.Module):
    def __init__(self,
                 vocab_size,
                 counter,
                 word_embed_size,
                 hidden_size,
                 n_layers,
                 bidirectional,
                 use_mlp,
                 dropout,
                 pad_index,
                 device,
                 inference):
        super(Model, self).__init__()
        self.vocab_size = vocab_size
        self.word_embed_size = word_embed_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.use_mlp = use_mlp
        self.device = device
        self.inference = inference
        self.rnn_output_size = hidden_size

        self.drop = nn.Dropout(dropout)
        self.l2r_emb = nn.Embedding(num_embeddings=vocab_size,
                                    embedding_dim=word_embed_size,
                                    pad_index=pad_index)
        self.l2r_rnn = nn.LSTM(input_size=word_embed_size,
                               hidden_size=hidden_size,
                               num_layers=n_layers,
                               batch_first=True)
        self.r2l_emb = nn.Embedding(num_embeddings=vocab_size,
                                    embedding_dim=word_embed_size,
                                    pad_index=pad_index)
        self.r2l_rnn = nn.LSTM(input_size=word_embed_size,
                               hidden_size=hidden_size,
                               num_layers=n_layers,
                               batch_first=True)

        self.init_weights()

    def init_weights(self):
        std = math.sqrt(1. / self.word_embed_size)
        self.l2r_emb.weight.data.normal_(0, std)
        self.r2l_emb.weight.data.normal_(0, std)

    def forward(self, sentences, target, target_pos=None):
        batch_size, seq_len = sentences.size()
        reversed_sentences = sentences.flip(1)[:, :-1]
