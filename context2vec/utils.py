import argparse
import numpy
from torchtext import data


def parse_args():

    gpu_id = -1
    parser = argparse.ArgumentParser(prog='src')
    parser.add_argument('--gpu-id', '-g', default=gpu_id, type=int)
    parser.add_argument('--train', '-t', action='store_true',
                        help='train or not')
    parser.add_argument('--input-file', '-i', default='dataset/sample.txt', type=str,
                        help='specify input file')
    parser.add_argument('--config-file', '-c', default='./config.toml', type=str,
                        help='specify config toml file')
    parser.add_argument('--wordsfile', '-w', default='models/embedding.vec',
                        help='word embeddings output filename')
    parser.add_argument('--modelfile', '-m', default='models/model.param',
                        help='model output filename')
    parser.add_argument('--task', default='', type=str,
                        help='choose evaluation task from [mscc]')

    return parser.parse_args()





class Dataset:
    def __init__(self,
                 sentences: list,
                 batch_size: int,
                 min_freq: int,
                 device: int,
                 pad_token='<PAD>',
                 unk_token='<UNK>',
                 bos_token='<BOS>',
                 eos_token='<EOS>',
                 seed=777):

        numpy.random.seed(seed)
        self.sent_dict = self._gathered_by_lengths(sentences)
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.device = device

        self.sentence_field = data.Field(use_vocab=True,
                                         unk_token=self.unk_token,
                                         pad_token=self.pad_token,
                                         init_token=self.bos_token,
                                         eos_token=self.eos_token,
                                         batch_first=True,
                                         include_lengths=False)
        self.sentence_id_field = data.Field(use_vocab=False, batch_first=True)

        self.sentence_field.build_vocab(sentences, min_freq=min_freq)
        self.vocab = self.sentence_field.vocab
        if self.pad_token:
            self.pad_index = self.sentence_field.vocab.stoi[self.pad_token]

        self.dataset = self._create_dataset(self.sent_dict, sentences)

    def get_raw_sentence(self, sentences):
        return [[self.vocab.itos[idx] for idx in sentence]
                for sentence in sentences]

    def _gathered_by_lengths(self, sentences):
        lengths = [(index, len(sent)) for index, sent in enumerate(sentences)]
        lengths = sorted(lengths, key=lambda x: x[1], reverse=True)

        sent_dict = dict()
        current_length = -1
        for (index, length) in lengths:
            if current_length == length:
                sent_dict[length].append(index)
            else:
                sent_dict[length] = [index]
                current_length = length

        return sent_dict

    def _create_dataset(self, sent_dict, sentences):
        datasets = dict()
        _fields = [('sentence', self.sentence_field),
                   ('id', self.sentence_id_field)]
        for sent_length, sent_indices in sent_dict.items():
            sent_indices = numpy.array(sent_indices)
            items = [*zip(sentences[sent_indices], sent_indices[:, numpy.newaxis])]
            datasets[sent_length] = data.Dataset(self._get_examples(items, _fields), _fields)
        return numpy.random.permutation(list(datasets.values()))

    def _get_examples(self, items: list, fields: list):
        return [data.Example.fromlist(item, fields) for item in items]

    def get_batch_iter(self, batch_size: int):

        def sort(data: data.Dataset) -> int:
            return len(getattr(data, 'sentence'))

        for dataset in self.dataset:
            yield data.Iterator(dataset=dataset,
                                batch_size=batch_size,
                                sort_key=sort,
                                train=True,
                                repeat=False,
                                device=self.device)

import toml


class Config:

    def __init__(self, filename: str):

        self.filename = filename
        config = toml.load(self.filename)

        nets = config.get('nets', {})
        self.word_embed_size = int(nets.get('word_embed_size', 300))
        self.hidden_size = int(nets.get('hidden_size', 300))
        self.n_layers = int(nets.get('n_layers', 1))
        self.use_mlp = bool(nets.get('use_mlp', True))
        self.dropout = float(nets.get('dropout', 0.0))

        train = config.get('train', {})
        self.n_epochs = int(train.get('n_epochs', 10))
        self.batch_size = int(train.get('batch_size', 100))
        self.min_freq = int(train.get('min_freq', 1))
        self.ns_power = float(train.get('ns_power', 0.75))
        self.learning_rate = float(train.get('learning_rate', 1e-4))

        mscc = config.get('mscc', {})
        self.question_file = mscc.get('question_file')
        self.answer_file = mscc.get('answer_file')


if __name__ == '__main__':
    config = Config('./config.toml')
    print(config)

import json


def write_embedding(id2word, nn_embedding, use_cuda, filename):
    with open(filename, mode='w') as f:
        f.write('{} {}\n'.format(nn_embedding.num_embeddings, nn_embedding.embedding_dim))
        if use_cuda:
            embeddings = nn_embedding.weight.data.cpu().numpy()
        else:
            embeddings = nn_embedding.weight.data.numpy()

        for word_id, vec in enumerate(embeddings):
            word = id2word[word_id]
            vec = ' '.join(list(map(str, vec)))
            f.write('{} {}\n'.format(word, vec))


def load_vocab(filename):
    with open(filename, mode='r') as f:
        f.readline()
        itos = [str(field.split(' ', 1)[0]) for field in f]
    stoi = {token: i for i, token in enumerate(itos)}
    return itos, stoi


def write_config(filename, **kwargs):
    with open(filename, mode='w') as f:
        json.dump(kwargs, f)


def read_config(filename):
    with open(filename, mode='r') as f:
        return json.load(f)