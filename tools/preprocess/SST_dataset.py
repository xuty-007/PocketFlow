import os
import torch
import torchtext.data as data
import argparse
import datetime
import numpy as np
import pickle

from tqdm import tqdm
from torchtext.datasets import IMDB, SST
from torchtext.vocab import GloVe

parser = argparse.ArgumentParser(description="Skip LSTM")
parser.add_argument('-batch-size', type=int, default=4096, help='batch size for training [default: 64]')
parser.add_argument('-save-interval', type=int, default=500, help='how many steps to wait before saving [default:500]')
parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
# data
parser.add_argument('-shuffle', action='store_true', default=False, help='shuffle the data every epoch')
# model
parser.add_argument('-dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
parser.add_argument('-large_cell_size', type=int, default=100, help='hidden size of large rnn cell [default: 100]')
parser.add_argument('-small_cell_size', type=int, default=5, help='hidden size of small rnn call [default 5]')
parser.add_argument('-num_layers', type=int, default=1, help='number of hidden layer [default 1]')
parser.add_argument('-embed_dim', type=int, default=300, help='number of embedding dimension [default: 128]')
parser.add_argument('-hidden_layer', type=int, default=200,
                    help='dimension of hidden layer in the fully connected network [default: 200]')
parser.add_argument('-gamma', type=float, default=0.01, help='gamma regularization parameter [default: 0.01]')
parser.add_argument('-tau', type=float, default=0.5, help='gamma regularization parameter [default: 0.5]')
# device
parser.add_argument('-device', type=int, default=0, help='device to use for iterate data, 0 mean cpu [default: -1]')
parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the gpu')
# option
parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
parser.add_argument('-predict', type=str, default=None, help='predict the sentence given')
parser.add_argument('-test', action='store_true', default=False, help='train or test')
args = parser.parse_args()

def load_data(text_field, label_field, **kwargs):
    train_data, test_data, _ = SST.splits(text_field, label_field,
                                          filter_pred=lambda ex: ex.label != 'neutral')
    text_field.build_vocab(train_data, vectors=GloVe()),
    label_field.build_vocab(train_data, test_data)
    train_iter, test_iter = data.BucketIterator.splits(
        (train_data, test_data),
        batch_sizes=(args.batch_size, args.batch_size),
        shuffle=args.shuffle,
        **kwargs
    )
    return train_iter, test_iter


print("\nLoading data...")
text_field = data.Field(batch_first=True, lower=True, tokenize='spacy')
label_field = data.Field(sequential=False, unk_token=None)
train_iter, test_iter = load_data(text_field, label_field, device=-1, repeat=False)

args.vocab_size = len(text_field.vocab)
args.n_class = len(label_field.vocab)
args.word_dict = text_field.vocab

args.cuda = (not args.no_cuda) and torch.cuda.is_available()
del args.no_cuda
args.save_dir = os.path.join(args.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

X = []
Y = []
for batch in tqdm(train_iter):
    x, y = batch.text, batch.label
    X.append(x)
    Y.append(y)

train_X = np.vstack((np.pad(X[0],(0,1), 'constant', constant_values=1),X[1]))
train_Y = np.hstack(Y)

X = []
Y = []
for batch in tqdm(test_iter):
    x, y = batch.text, batch.label
    X.append(x)
    Y.append(y)

test_X = np.pad(X[0], (0, 4), 'constant', constant_values=1)
test_Y = Y[0]

pickle.dump((train_X, train_Y), open(".data/sst/SST_train.pickle",'wb'))
pickle.dump((test_X, test_Y), open(".data/sst/SST_test.pickle", 'wb'))

