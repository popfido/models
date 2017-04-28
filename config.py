import argparse


class ConfigHolder(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='DOCUMENT VECTOR estimation toolkit',
                                              formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.parser.add_argument('train', metavar='TRAIN_FILE', type=str, help='Text data used to train the model')
        self.parser.add_argument('-test', metavar='TEST_FILE', type=str, help='Text data used to test the model')
        self.parser.add_argument('-size', metavar='WV_SIZE', dest='embed_dim', type=int, help='Size of word vectors',
                                 default=200)
        self.parser.add_argument('-window', metavar='WINDOW_SIZE', type=int, help='Max skip length between words',
                                 default=5)
        self.parser.add_argument('-sample', metavar='SAMPLE_THRESHOLD', type=float, default=1e-3,
                                 help='Threshold for occurrence of words. Those that appear with higher frequency in '
                                      'the training data will be randomly down-sampled; useful range is (0, 1e-5)', )
        self.parser.add_argument('-num_neg', type=int, default=5,
                                 help='Number of negative examples; common values are 3 - 10 (0 = not used)')
        self.parser.add_argument('-hs', '--use_hs', action='store_false', help='Use Hierarchical Softmax')
        self.parser.add_argument('-word', metavar='WORD2VEC_FILE', type=str,
                                 help='File used to save the resulting word vectors', default='word_vector.dat')
        self.parser.add_argument('-output', metavar='DOC2VEC_FILE', type=str,
                                 help='File used to save the resulting document vectors', default='doc_vector.dat')
        self.parser.add_argument('-threads', metavar='NUM_THREADS', type=int, default=0,
                                 help='Number of CPU threads to compute if use CPU; 0 means using all CPU threads')
        self.parser.add_argument('-e', '-epoch', metavar='NUM_ITERATION', dest='epoch', type=int,
                                 help='Number of training iteration', default=20)
        self.parser.add_argument('-min_count', type=int, dest='countThreshold', default=10,
                                 help='This will discard words that appear less than <int> times')
        self.parser.add_argument('-alpha', metavar='LEARNING_RATE', type=float, default=0.025,
                                 help='The starting learning rate; default is 0.025 for skip-gram and 0.05 for CBOW')
        self.parser.add_argument('-debug', metavar='DEBUG_LEVEL', type=int, default=2,
                                 help='the debug mode level; 2 = more info during training')
        self.parser.add_argument('-binary', '--binary_saved', action='store_false',
                                 help='Save the resulting vectors in binary moded')
        self.parser.add_argument('-sv', '--save-vocab', metavar='VOCAB_FILE', type=str, dest='saveVocab',
                                 help='The vocabulary will be saved to <file>', default='')
        self.parser.add_argument('-rv', '--read-vocab', metavar='VOCAB_FILE', type=str, dest='vocab', default='',
                                 help='The vocabulary will be read from <file>, not constructed from the training data')
        self.parser.add_argument('-cbow', metavar='MODE', type=int,
                                 help='Whether use the continuous bag of words model', default=1)
        self.parser.add_argument('-ss', '-sentence-sample', metavar='WORD_SAMPLE_RATE', type=float, default=0.1,
                                 help='The rate to sample words out of a document for documenet representation')
        self.parser.add_argument('-stat_interval', metavar='INTERVAL', type=int,
                                 help='How often to print statistics(iteration)', default=1000)
        self.parser.add_argument('-check_interval', metavar='INTERVAL', type=int,
                                 help='How often to write checkpoints(iteration)', default=10000)
        self.parser.add_argument('-concat', action='store_true', help='whether concat word vecs in paravec')
        self.parser.add_argument('-batch_size', metavar='BATCH_SIZE', type=int, help='Minibatch size', default=100)
        self.parser.add_argument('-train_method', metavar='ALGO', type=str,
                                 help='Optimization Algorithm', default='Adam')
        self.parser.add_argument('-loss', metavar='FUNC_NAME', type=str,
                                 help='Loss function of embedding model', default='nce')
        self.parser.add_argument('--version', action='version', version='%(prog)s 2.0')

    def init(self):
        for name, val in vars(self.parser.parse_args()).items():
            setattr(self, name, val)

config_holder = ConfigHolder()
