import codecs
import numpy as np
import os
from config import config_holder
from collections import Counter
from itertools import dropwhile

vocab = {}

def readVocab(vocab_path):
    assert(os.path.exists(vocab_path))
    assert(os.path.isfile(vocab_path))
    vocab = Counter()
    with codecs.open(vocab_path, encoding='utf-8') as f:
        for line in f:
            line = line.split()
            vocab[line[0]] = line[1]
    return vocab

def learnVocab(data_path, debug=0):
    assert(os.path.exists(data_path))
    assert(os.path.isfile(data_path))
    vocab = Counter()
    with codecs.open(data_path, encoding='utf-8') as f:
        for line in f:
            vocab += Counter(line.split())

    # Reduces the vocabulary by removing infrequent tokens
    for key, count in dropwhile(lambda key_count: key_count[1] >= config_holder.countThreshold, vocab.most_common()):
        del vocab[key]

    if (debug > 0):
        print("Vocab size: {0}".format(len(vocab)))
        print("Words in train file: {0}".format(sum(vocab.values())))

    return vocab

def saveVocab(save_path, dic):
    with codecs.open(save_path, mode='w', encoding='utf-8') as f:
        written = u''
        for k, v in dic.items():
            written = '\n'.join([written, '\t'.join([k,str(v)])])
        f.write(written)
        f.flush()

if __name__ == '__main__':
    config_holder.init()
    data_path = config_holder.train
    read_vocab_path = config_holder.vocab
    save_vocab_path = config_holder.saveVocab
    if len(read_vocab_path) > 0:
        vocab = readVocab(read_vocab_path)
    else:
        vocab = learnVocab(data_path)

    if len(save_vocab_path) > 0:
        saveVocab(save_vocab_path, vocab)

    
