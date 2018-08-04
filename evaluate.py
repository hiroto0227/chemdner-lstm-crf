import os
import utils
from utils import *
import pandas as pd
import chemdner_data_converter


def labels_to_spantokens(df, mode='label'):
    """dfを受け取ってlabel結果からspantokenにして返す。
    mode : label or pred_label
    """
    spantokens = set()
    entity = ''
    ann_start = 0
    pre_label = utils.SOS
    for key, row in df.iterrows():
        now_label = row[mode]
        pre_now = (pre_label, now_label)
         # pre_labelとnow_labelのあり得る組み合わせの中で分岐(SOSやBOSはここで弾く。)
        if pre_now in [(B, M), (B, E), (M, E), (E, B), (E, S), (S, B), (O, S), (O, B), (O, M), (O, E)]:
            if now_label == S:
                spantokens.add((row.token, row.start_ix, row.end_ix))
            elif now_label == B:
                ann_start = row.start_ix
                entity += row.token
            elif now_label == M:
                entity += row.token
            elif now_label == E:
                entity += row.token
                spantokens.add((entity, ann_start, row.end_ix))
                entity = ''
            elif now_label == O:
                entity += row.token
        pre_label = now_label
    return spantokens

if __name__ == '__main__':
    CHEM_DATA_PATH = '/Users/user/chemdner_pytorch/chemdner_datas/'
    correct = 0
    test_num = 0
    pred_num = 0
    # load data
    pred_df = pd.read_csv(os.path.join(CHEM_DATA_PATH, 'pred.csv'))
    for file_ix in set(pred_df.file_ix):
        test_spantokens = labels_to_spantokens(pred_df[pred_df.file_ix == file_ix], mode="label")
        pred_spantokens = labels_to_spantokens(pred_df[pred_df.file_ix == file_ix], mode="pred_label")
        print("-----------------")
        print(file_ix)
        print(test_spantokens)
        print(pred_spantokens)
        test_num += len(test_spantokens)
        pred_num += len(pred_spantokens)
        for test_spantoken in test_spantokens:
            for pred_spantoken in pred_spantokens:
                if test_spantoken == pred_spantoken:
                    correct += 1
    print("test_num: {}, pred_num: {}, correct_num: {}".format(test_num, pred_num, correct))
    print("precision : {}".format(correct / pred_num))
    print("recall : {}".format(correct / test_num))
    precision = correct / pred_num
    recall = correct / test_num
    print("f-score : {}".format((2 * precision * recall) / (precision + recall)))