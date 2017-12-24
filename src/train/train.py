import argparse
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .data import parse_corpus, format_data
from .model import Net


def load_data(path, seq_length, batch_size):
    dataX, dataY, char_to_int, int_to_char, chars = parse_corpus(path, seq_length=seq_length)
    data = format_data(dataX, dataY, n_classes=len(chars), batch_size=batch_size)

    return data, dataX, dataY, char_to_int, int_to_char, chars

def save_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def train(model, optimizer, epoch, data, log_interval):
    model.train()

    for batch_i, (seq_in, target) in enumerate(data):
        seq_in, target = Variable(seq_in), Variable(target)
        optimizer.zero_grad()

        output = model(seq_in)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        # Log training status
        if batch_i % log_interval == 0:
            print('Train epoch: {} ({:2.0f}%)\tLoss: {:.6f}'.format(epoch, 100. * batch_i / len(data), loss.data[0]))

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train seq2seq model')
    parser.add_argument('corpus', type=str, metavar='F',
                        help='training corpus file')
    parser.add_argument('--seq-length', type=int, default=50, metavar='N',
                        help='input sequence length (default: 50)')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='training batch size (default: 1)')
    parser.add_argument('--embedding-dim', type=int, default=128, metavar='N',
                        help='embedding dimension for characters in corpus (default: 128)')
    parser.add_argument('--hidden-dim', type=int, default=64, metavar='N',
                        help='hidden state dimension (default: 64)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--dropout', type=float, default=0.2, metavar='DR',
                        help='dropout rate (default: 0.2)')
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (default: 30)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='number of batches to wait before logging status (default: 10)')
    parser.add_argument('--save-interval', type=int, default=10, metavar='N',
                        help='number of epochs to wait before saving model (default: 10)')
    parser.add_argument('--output', type=str, default='model.bin', metavar='F',
                        help='output model file')
    parser.add_argument('--output-c', type=str, default='corpus.bin', metavar='F',
                        help='output corpus related file (mappings & vocab)')
    args = parser.parse_args()

    # Prepare
    train_data, dataX, dataY, char_to_int, int_to_char, chars = load_data(args.corpus, seq_length=args.seq_length, batch_size=args.batch_size)
    model = Net(len(chars), args.embedding_dim, args.hidden_dim, dropout=args.dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Train
    for epoch in range(args.epochs):
        train(model, optimizer, epoch, train_data, log_interval=args.log_interval)

        if (epoch + 1) % args.save_interval == 0:
            model.eval()
            torch.save(model, args.output)

    # Save mappings, vocabs, & model
    save_pickle((dataX, char_to_int, int_to_char, chars), args.output_c)

    model.eval()
    torch.save(model, args.output)
