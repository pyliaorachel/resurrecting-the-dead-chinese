import argparse
import random


parser = argparse.ArgumentParser(description='Mix files')
parser.add_argument('file1', type=str,
                    help='First filename')
parser.add_argument('file2', type=str,
                    help='Second filename')
parser.add_argument('--output', type=str,
                    help='Output filename')

args = parser.parse_args()


# Read lines from files
with open(args.file1, 'rb') as f1:
    ls1 = f1.readlines()
with open(args.file2, 'rb') as f2:
    ls2 = f2.readlines()

# Shuffle
ls = ls1 + ls2
random.shuffle(ls)

# Write to output file
with open(args.output, 'wb') as fo:
    fo.writelines(ls)
