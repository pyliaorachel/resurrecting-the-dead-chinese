import argparse
import pickle

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable


def load_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def is_end(c):
    end_tokens = ['。', '？', '！', '.', '?', '!']
    return c in end_tokens

def to_prob(vec):
    s = sum(vec)
    return [v / s for v in vec]

def gen_text(model, patterns, char_to_int, int_to_char, chars, n_sent=10, restart_seq=False):
    n_patterns = len(patterns)

    # Randomly choose a pattern to start text generation
    start = np.random.randint(0, n_patterns - 1)
    pattern = patterns[start]

    # Start generation until n_sent sentences generated 
    cnt = 0
    while cnt < n_sent: 
        # Format input pattern
        seq_in = np.array(pattern)
        seq_in = seq_in.reshape(1, -1) # batch_size = 1

        seq_in = Variable(torch.LongTensor(seq_in))

        # Predict next character
        pred = model(seq_in)
        pred = to_prob(F.softmax(pred, dim=1).data[0].numpy()) # turn into probability distribution
        char = np.random.choice(chars, p=pred)                 # pick char based on probability instead of always picking the highest value
        char_idx = char_to_int[char]
        print(char, end='')

        # Append predicted character to pattern, truncate to usual pattern size, use as new pattern
        pattern.append(char_idx)
        pattern = pattern[1:]

        if is_end(char):
            if restart_seq:
                start = np.random.randint(0, n_patterns - 1)
                pattern = patterns[start]
                print()

            cnt += 1 
    
    if not restart_seq:
        print()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate text')
    parser.add_argument('corpus', type=str, metavar='F',
                        help='corpus-related data file')
    parser.add_argument('model', type=str, metavar='F',
                        help='model for text generation')
    parser.add_argument('--n-sent', type=int, default=10, metavar='N',
                        help='number of sentences to generate (default: 10)')
    parser.add_argument('--restart-seq', action='store_true', 
                        help='whether to randomly pick a new sequence to start the next sentence generation (default: F)')

    args = parser.parse_args()

    # Load mappings & vocabularies
    dataX, char_to_int, int_to_char, chars = load_pickle(args.corpus)

    # Load model
    model = torch.load(args.model)
    
    # Generate text
    gen_text(model, dataX, char_to_int, int_to_char, chars, n_sent=args.n_sent, restart_seq=args.restart_seq)
