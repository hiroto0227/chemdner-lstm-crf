import csv, os
import pandas as pd
import utils


CHEM_DATA_PATH = '/Users/user/chemdner_pytorch/chemdner_datas/'


def load_token_to_id():
    with open(os.path.join(CHEM_DATA_PATH, 'token2ix')) as f:
        token2ix = {token: i for i, token in enumerate(f.read().split('\n'))}
    token2ix.update({utils.PAD: len(token2ix), utils.UNK: len(token2ix) + 1})
    return token2ix


def load_label_to_id():
    with open(os.path.join(CHEM_DATA_PATH, 'label2ix')) as f:
        label2ix = {label: i for i, label in enumerate(f.read().split('\n'))}
    label2ix.update({utils.PAD: len(label2ix), utils.UNK: len(label2ix) + 1})
    return label2ix