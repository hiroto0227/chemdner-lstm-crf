import utils
import torch
from torch.autograd import Variable as Var


def to_vector_by_df(df, token2ix, label2ix, batch_size=None):
    X = []
    Y = []
    for file_ix in set(df.file_ix):
        token_ixs = [token2ix.get(token, token2ix[utils.UNK]) for token in df[df.file_ix == file_ix].token]
        X.extend(batching(token_ixs, batch_size=batch_size, pad_ix=token2ix[utils.PAD]))
        label_ixs = [label2ix.get(label, None) for label in df[df.file_ix == file_ix].label]
        Y.extend(batching(label_ixs, batch_size=batch_size, pad_ix=label2ix[utils.PAD]))
    return Var(torch.LongTensor(X)), Var(torch.LongTensor(Y))


def batching(seq, batch_size, pad_ix):
    """seq(1次元配列)を受け取って、batch_sizeに分割して返す。(余りはパディング。)"""
    batched_seq = []
    _seq = []
    for s in seq:
        _seq.append(s)
        if len(_seq) >= batch_size:
            batched_seq.append(_seq)
            _seq = []
    pad_length = batch_size - len(_seq)
    batched_seq.append(_seq + [pad_ix] * pad_length)
    return batched_seq