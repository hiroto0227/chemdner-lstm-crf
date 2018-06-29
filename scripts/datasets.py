import sys
import os
sys.path.extend(['../', './scripts/'])
import config
from tqdm import tqdm
from transformer import annfile2annset


def short():
    texts = []
    anns = []
    file_ids = [f[:-4] for f in os.listdir(config.short_data_root) if f[-4:] == '.txt']
    for file_id in tqdm(file_ids):
        with open(config.short_data_root + '{}.txt'.format(file_id)) as f:
            texts.append(f.read())
        anns.append(annfile2annset(config.short_data_root + '{}.ann'.format(file_id)))
    assert len(texts) == len(anns), 'must have the same length texts and anns'
    return texts, anns


def train():
    texts = []
    anns = []
    file_ids = [f[:-4] for f in os.listdir(config.train_data_root) if f[-4:] == '.txt']
    for file_id in tqdm(file_ids):
        with open(config.train_data_root + '{}.txt'.format(file_id)) as f:
            texts.append(f.read())
        anns.append(annfile2annset(config.train_data_root + '{}.ann'.format(file_id)))
    assert len(texts) == len(anns), 'must have the same length texts and anns'
    return texts, anns


def valid():
    texts = []
    anns = []
    file_ids = [f[:-4] for f in os.listdir(config.valid_data_root) if f[-4:] == '.txt']
    for file_id in tqdm(file_ids):
        with open(config.valid_data_root + '{}.txt'.format(file_id)) as f:
            texts.append(f.read())
        anns.append(annfile2annset(config.valid_data_root + '{}.ann'.format(file_id)))
    assert len(texts) == len(anns), 'must have the same length texts and anns'
    return texts, anns


def test():
    texts = []
    anns = []
    file_ids = [f[:-4] for f in os.listdir(config.test_data_root) if f[-4:] == '.txt']
    for file_id in tqdm(file_ids):
        with open(config.test_data_root + '{}.txt'.format(file_id)) as f:
            texts.append(f.read())
        anns.append(annfile2annset(config.test_data_root + '{}.ann'.format(file_id)))
    assert len(texts) == len(anns), 'must have the same length texts and anns'
    return texts, anns
