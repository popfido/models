import re
import bs4
import os
import pickle
import codecs
from tqdm import tqdm
from urllib.request import urlretrieve
import zipfile
import tensorflow as tf
from distutils.version import LooseVersion
import warnings


class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num


def load_text8():
    dataset_folder_path = 'data'
    dataset_filename = 'text8.zip'
    dataset_name = 'Text8 Dataset'

    if not os.path.isfile(dataset_filename):
        with DLProgress(unit='B', unit_scale=True, miniters=1, desc=dataset_name) as pbar:
            urlretrieve(
                'http://mattmahoney.net/dc/text8.zip',
                dataset_filename,
                pbar.hook)

    if not os.path.isdir(dataset_folder_path):
        with zipfile.ZipFile(dataset_filename) as zip_ref:
            zip_ref.extractall(dataset_folder_path)

    return load_data(os.path.join(dataset_folder_path, dataset_filename.split('.')[0]))


def load_data(path):
    """
    load dataset from file
    """
    assert(os.path.exists(path))
    assert(os.path.isfile(path))
    input_file = path
    with codecs.open(input_file, "r") as f:
        data = f.read()

    return data


def load_preprocess():
    """
    Load the Preprocessed Training data and return them in batches of <batch_size> or less
    """
    return pickle.load(open('preprocess.p', mode='rb'))


def save_params(params):
    """
    Save parameters to file
    """
    pickle.dump(params, open('params.p', 'wb'))


def load_params():
    """
    Load parameters from file
    """
    return pickle.load(open('params.p', mode='rb'))


def isFullEnglish(sentences):
    text = ' '.join(sentences)
    text_rm_eng = re.sub(r'[a-zA-Z]+', '', text)
    return True if len(text_rm_eng) * 1.0 / len(txt) < 0.9 else False


def token_lookup(tokenize=True):
    return {
        '.': ' ||period|| ',
        ',': ' ||comma|| ',
        '"': ' ||quotation_mark|| ',
        ';': ' ||semicolon|| ',
        '!': ' ||exclamation_mark|| ',
        '?': ' ||question_mark|| ',
        '(': ' ||left_parentheses|| ',
        ')': ' ||right_parentheses|| ',
        '--': ' ||dash|| ',
        '\n': ' ||return|| '
    } if tokenize else {
        '.': '',
        ',': '',
        '"': '',
        ';': '',
        '!': '',
        '?': '',
        '(': '',
        ')': '',
        '--': '',
        '\n': ''
    }


def check_tf_version():
    assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer'
    print('TensorFlow Version: {}'.format(tf.__version__))
    if not tf.test.gpu_device_name():
        warnings.warn('No GPU found. Please use a GPU to train your neural network.')
    else:
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def clean_html(sentences):
    return sentences
