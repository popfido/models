# coding=utf-8

import re
from bs4 import BeautifulSoup
import os
import pickle
import codecs
from tqdm import tqdm
from urllib.request import urlretrieve
import zipfile


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


def isFullEnglish(sentence):
    text = ' '.join(sentence)
    text_rm_eng = re.sub(r'[a-zA-Z]+', '', text)
    return True if len(text_rm_eng) * 1.0 / len(sentence) < 0.9 else False


def token_lookup(tokenize=True):
    return {
        '.': ' ||period|| ',
        '。': ' ||period|| ',
        ',': ' ||comma|| ',
        '，': ' ||comma|| ',
        '"': ' ||quotation_mark|| ',
        ';': ' ||semicolon|| ',
        '；': ' ||semiccolon|| ',
        '!': ' ||exclamation_mark|| ',
        '！': ' ||exclamation_mark|| ',
        '?': ' ||question_mark|| ',
        '？': ' ||question_mark|| ',
        '(': ' ||left_parentheses|| ',
        '（': ' ||left_parentheses|| ',
        ')': ' ||right_parentheses|| ',
        '）': ' ||right_parentheses|| ',
        '--': ' ||dash|| ',
        '——': ' ||dash|| ',
        '\n': ' ||return|| '
    } if tokenize else {
        '.': '',
        '。': '',
        ',': '',
        '，': '',
        '"': '',
        ';': '',
        '；': '',
        '!': '',
        '！': '',
        '?': '',
        '？': '',
        '(': '',
        '（': '',
        ')': '',
        '）': '',
        '--': '',
        '——': '',
        '\n': ''
    }


def clean_html(sentences):
    soup = BeautifulSoup(sentences)
    cleaned = re.sub('<[^<]+?>|([0-9]、)|^\n|(\xa0)', '', soup.get_text())
    return cleaned
