import os, csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable as Var
import utils
from model import LSTMCRF_Model, MyLSTM


TRAIN_FILE_DIR = '{}/datas/train/'.format(os.path.dirname(os.path.realpath(__file__)))


def load_data():
    tokens_list = []
    labels_list = []
    train_files = [file for file in os.listdir(TRAIN_FILE_DIR) if file.endswith('.csv')]
    for train_file in train_files:
        tokens = []
        labels = []
        with open(TRAIN_FILE_DIR + train_file, 'rt') as f:
            csv_reader = csv.reader(f)
            for token, label in csv_reader:
                tokens.append(token)
                labels.append(label)
        tokens_list.append(tokens)
        labels_list.append(labels)
    return tokens_list, labels_list


def load_token_to_id():
    with open(TRAIN_FILE_DIR + 'token_to_id') as f:
        id_to_token = f.read().split('\n')
    return {token: i for i, token in enumerate(id_to_token)}


def load_label_to_id():
    with open(TRAIN_FILE_DIR + 'label_to_id') as f:
        id_to_label = f.read().split('\n')
    return {label: i for i, label in enumerate(id_to_label)}


def make_x(tokens_list, token_to_id, batch_size=32):
    """
    :param tokens_list: (file_nums, text_token_length)
    :param token_to_id: dict
    :param batch_size: int (2, 4, 8, 16...512)
    :return: (seq_length, batch_size)
    """
    X = []
    for tokens in tokens_list:
        batch_x = [token_to_id.get(utils.SOS)]
        for token in tokens:
            batch_x.append(token_to_id.get(token, token_to_id[utils.UNK]))
            if len(batch_x) >= batch_size:
                X.append(batch_x)
                batch_x = []
        batch_x.append(token_to_id.get(utils.EOS))
        pad_length = batch_size - len(batch_x)
        X.append(batch_x + [token_to_id.get(utils.PAD)] * pad_length)
    return Var(torch.LongTensor(X))


def make_y(labels_list, label_to_id, batch_size=36):
    Y = []
    for labels in labels_list:
        batch_y = [label_to_id.get(utils.SOS)]
        for label in labels:
            batch_y.append(label_to_id.get(label))
            if len(batch_y) >= batch_size:
                Y.append(batch_y)
                batch_y = []
        batch_y.append(label_to_id.get(utils.EOS))
        pad_length = batch_size - len(batch_y)
        Y.append(batch_y + [label_to_id.get(utils.PAD)] * pad_length)
    return Var(torch.LongTensor(Y))


def train():
    EPOCH = 10
    BATCH_SIZE = 512
    tokens_list, labels_list = load_data()
    token_to_id = load_token_to_id()
    label_to_id = load_label_to_id()
    X = make_x(tokens_list, token_to_id, batch_size=BATCH_SIZE)
    Y = make_y(labels_list, label_to_id, batch_size=BATCH_SIZE)
    model = MyLSTM(len(token_to_id), len(label_to_id))
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    for i in range(EPOCH):
        try:
            loss_sum = 0
            for x, y in zip(X, Y):
                model.zero_grad()
                tag_scores = model(x)
                loss = loss_function(tag_scores, y)
                loss.backward()
                print(loss)
                optimizer.step()
                loss_sum += float(loss)
            print('{}epoch --- loss: {}'.format(i, loss))
        except KeyboardInterrupt:
            torch.save(model, '../outsource/mymodel.pth')


if __name__ == '__main__':
    train()