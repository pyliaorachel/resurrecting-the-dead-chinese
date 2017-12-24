# Resurrecting the Dead - Chinese

Text generation system based on a mixed corpus of 《毛澤東語錄》(Quotations From Chairman Mao Tse-Tung) and《論語》(Confucian Analects).

|Framework|Model|Optimizer|
|:-:|:-:|:-:|
| PyTorch | RNN (LSTM) | Adam |

[穿越時空的偉人：用PyTorch重現偉人們的神經網絡](https://pyliaorachel.github.io/blog/tech/nlp/2017/12/24/resurrecting-the-dead-chinese.html)

## Usage

###### Mix corpus

```bash
$ cd src/corpus
$ python3 mix.py <first-corpus> <second-corpus> --output <output-corpus-text-file>
```

###### Train

```bash
$ cd src
$ python3 -m train.train <corpus-text-file> 

# For more options
$ python3 -m train.train -h
```

Outputs:

- `model.bin`: torch model
- `corpus.bin`: parsed corpus, mapping, & vocabulary

###### Text generation

```bash
$ cd src
$ python3 -m generate_text.gen <corpus-bin-file> <model-bin-file>

# For more options
$ python3 -m generate_text.gen -h
```

## Structure

```
├── corpus                                          # Raw & parsed corpus
│   ├── corpus.txt                                      # Main corpus file for training
│   ├── luen_yu_clean.txt                               # Raw corpus with irrelevant words removed
│   ├── luen_yu_raw.txt                                 # Raw corpus
│   ├── luen_yu_sent.txt                                # Clean corpus seperated into sentences
│   ├── mao_clean.txt                                   # Raw corpus with irrelevant words removed
│   ├── mao_raw.txt                                     # Raw corpus
│   └── mao_sent.txt                                    # Clean corpus seperated into sentences
├── output                                          # Results
│   ├── log                                             # Log files
│   └── model                                           # Pretrained models
│       └── slxx-bsxx-edxx-hdxx-lrxx-drxx-epxx              # seq_length, batch_size, embedding_dim, hidden_dim, 
│                                                           # learning_rate, dropout, epochs
└── src                                             # Source codes
    ├── corpus                                          # Corpus processing
    │   ├── mix.py                                          # Mix two corpora
    │   └── run.sh                                          # Running the script
    ├── generate_text                                   # Text generation
    │   └── gen.py                                          # Text generation
    ├── train                                           # Model training
    │   ├── data.py                                         # Parse data
    │   ├── model.py                                        # Main LSTM model
    │   └── train.py                                        # Training
    ├── gen.sh                                          # Running text generation script
    └── train.sh                                        # Running training script
```
