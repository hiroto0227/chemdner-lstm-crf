import os, csv, sys
import pandas as pd
import dataset
import transformer
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable as Var
import utils
from model import MyLSTM


if __name__ == '__main__':
    CHEM_DATA_PATH = '/Users/user/chemdner_pytorch/chemdner_datas/'
    MODEL_PATH = "./outsource/mymodel.pth"
    EPOCH = 100
    BATCH_SIZE = 32
    # prepare data
    train_df = pd.read_csv(os.path.join(CHEM_DATA_PATH, 'train.csv'))[:3000]
    token2ix = dataset.load_token_to_id()
    label2ix = dataset.load_label_to_id()
    # transform
    X, Y = transformer.to_vector_by_df(train_df, token2ix, label2ix, BATCH_SIZE)
    print(X.size(), Y.size())
    assert X.size() == Y.size(), "XとYのサイズが違います。"
    # train model
    model = MyLSTM(vocab_size=len(token2ix), tag_size=len(label2ix), BATCH_SIZE=BATCH_SIZE)
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
                optimizer.step()
                loss_sum += float(loss)
            print('{}epoch --- loss: {}'.format(i, loss_sum))
        except KeyboardInterrupt:
            print('model saved!!')
            torch.save(model.state_dict(), MODEL_PATH)
            sys.exit(1)
    print('model saved!!')
    torch.save(model.state_dict(), MODEL_PATH)