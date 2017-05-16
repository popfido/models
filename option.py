# coding=utf-8

class Option(object):
    """
    Options used by embedding model.
    """

    def __init__(self, config_holder, vocab):
        # Model options.

        self.model = "doc2vecc"

        # Embedding dimension.
        self.embed_dim = config_holder.embed_dim

        # Window size
        self.window_size = config_holder.window

        # Number of negative samples per example.
        self.num_samples = config_holder.num_neg

        # The initial learning rate.
        self.learning_rate = config_holder.alpha

        # Number of epochs to train. After these many epochs, the learning
        # rate decays linearly to zero and the training stops.
        self.epochs_to_train = config_holder.epoch

        # Number of negative sample
        self.negative_sample_size = config_holder.num_neg

        # Concurrent training steps.
        # self.concurrent_steps = config_holder.concurrent_steps

        # Number of examples for one training step.
        self.batch_size = config_holder.batch_size

        # The minimum number of word occurrences for it to be included in the
        # vocabulary.
        self.min_count = config_holder.countThreshold

        # Training algorithm
        self.train_method = config_holder.train_method

        # Subsampling threshold for word occurrence.
        self.subsample = config_holder.sample

        # Vocabulary size
        self.vocab_size = len(vocab)

        # How often to print statistics.
        self.statistics_interval = config_holder.stat_interval

        # How often to write checkpoints (rounds up to the nearest statistics
        # interval).
        self.checkpoint_interval = config_holder.check_interval

        # Which of the loss the embedding model use
        self.loss = config_holder.loss

        self.concat = config_holder.concat

        self.sentence_sample = config_holder.ss if self.model != "doc2vecc" else 10

        self.dp_ratio = 0.5
