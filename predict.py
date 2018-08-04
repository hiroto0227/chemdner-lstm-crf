import os
import dataset
import transformer
import utils
import torch
import model
import pandas as pd
import chemdner_data_converter


if __name__ == '__main__':
    CHEM_DATA_PATH = '/Users/user/chemdner_pytorch/chemdner_datas/'
    MODEL_PATH = "./outsource/mymodel.pth"
    BATCH_SIZE = 32
    # prepare data
    test_df = pd.read_csv(os.path.join(CHEM_DATA_PATH, 'test.csv'))[:3000]
    token2ix = dataset.load_token_to_id()
    label2ix = dataset.load_label_to_id()
    ix2label = [k for k, v in label2ix.items()]
    # transform
    X, Y = transformer.to_vector_by_df(test_df, token2ix, label2ix, BATCH_SIZE)
    print(X.size(), Y.size())
    assert X.size() == Y.size(), "XとYのサイズが違います。"
    # load model
    model = model.MyLSTM(len(token2ix), len(label2ix), BATCH_SIZE=BATCH_SIZE)
    model.load_state_dict(torch.load(MODEL_PATH))
    pred_labels = [] 
    for x in X:
        tag_scores = model(x)
        values, max_ixs = torch.max(tag_scores, 1)
        # x == paddingの時はyをdecodeしない。
        pred_labels.extend([ix2label[int(y_ix)] for x_ix, y_ix in zip(x, max_ixs) if x_ix != token2ix[utils.PAD]])
    test_df['pred_label'] = pred_labels
    test_df.to_csv(os.path.join(CHEM_DATA_PATH, 'pred.csv'))