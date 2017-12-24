import numpy as np
import torch


def parse_corpus(path, seq_length=50):
    '''Parse raw corpus text into input-output pairs, where input is a sequence of characters, output is 1 character after the sequence'''

    # Read text
    with open(path, 'r') as f:
        raw_text = f.read().replace('\n', '')

    # Get unique characters
    chars = sorted(list(set(raw_text)))

    # Map char to int / int to char
    char_to_int = dict((c, i) for i, c in enumerate(chars))
    int_to_char = dict((i, c) for i, c in enumerate(chars))
    
    # Prepare training data, for every <seq_length> chars, predict 1 char after the sequence
    n_chars = len(raw_text)
    dataX = [] # N x seq_length
    dataY = [] # N x 1
    for i in range(0, n_chars - seq_length):
        seq_in = raw_text[i:i + seq_length]
        seq_out = raw_text[i + seq_length]
        dataX.append([char_to_int[char] for char in seq_in])
        dataY.append(char_to_int[seq_out])
    
    return (dataX, dataY, char_to_int, int_to_char, chars)

def format_data(dataX, dataY, n_classes, batch_size=64):
    '''Parse into minibatches, return Tensors'''

    # For simplicity, discard trailing data not fitting into batch_size
    n_patterns = len(dataY)
    n_patterns = n_patterns - n_patterns % batch_size
    X = dataX[:n_patterns]
    Y = dataY[:n_patterns]

    # Parse X
    X = np.array(X)
    _, seq_length = X.shape
    X = X.reshape(-1, batch_size, seq_length)

    X = torch.LongTensor(X)

    # Parse Y
    Y = np.array(Y)
    Y = Y.reshape(-1, batch_size)

    Y = torch.LongTensor(Y)

    return list(zip(X, Y))
