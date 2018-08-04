import os, sys, re, csv
import pandas as pd
from utils import S, B, M, E, O, SOS, EOS

CHEMDNER_STANDOFF_DIR = "/Users/user/chemdner/storage/datas/"
OUT_DIR_PATH = "/Users/user/chemdner_pytorch/chemdner_datas/"

def make_chemdner_data(mode='train'):
    # init
    fileid_col = []
    start_ix_col = []
    end_ix_col = []
    token_col = []
    label_col = []
    # data convert
    fileids = [filename.replace('.txt', '') 
        for filename in  os.listdir(os.path.join(CHEMDNER_STANDOFF_DIR, mode)) if filename.endswith('.txt')]
    for fileid in fileids:
        with open(os.path.join(CHEMDNER_STANDOFF_DIR, mode, fileid + '.txt'), 'rt') as f:
            text = f.read()
        with open(os.path.join(CHEMDNER_STANDOFF_DIR, mode, fileid + '.ann'), 'rt') as f:
            annotations = f.read().split('\n')
            ann_spantokens = annotations_to_spantokens(annotations)
        spantokens = text_to_spantokens(text)
        # update (SOSとEOSを入れる。)
        fileid_col.extend([fileid for i in range(len(spantokens) + 2)])
        token_col.extend([SOS] + [s[0] for s in spantokens] + [EOS])
        start_ix_col.extend([0] + [s[1] for s in spantokens] + [spantokens[-1][1]+1])
        end_ix_col.extend([0] + [s[2] for s in spantokens] + [spantokens[-1][2]+1])
        label_col.extend([SOS] + text_to_labels(text, ann_spantokens) + [EOS])
    df = pd.DataFrame({
        'file_ix': fileid_col,
        'start_ix': start_ix_col,
        'end_ix': end_ix_col,
        'token': token_col,
        'label': label_col
    }).to_csv(os.path.join(OUT_DIR_PATH, mode + '.csv'))
    # save vacab and tags (trainの場合のみ。)
    if mode == 'train':
        with open(os.path.join(OUT_DIR_PATH, 'token2ix'), 'wt') as f:
            f.write('\n'.join([token for token in sorted(set(token_col))]))
        with open(os.path.join(OUT_DIR_PATH, 'label2ix'), 'wt') as f:
            f.write('\n'.join([label for label in sorted(set(label_col))]))

def to_annfiles(csv_file, out_dir):
    df = pd.read_csv(csv_file)
    for fileid in df.fileids:
        spantokens = labels_to_spantokens(df[df.fileid == fileid].label)
        os.mkdir(out_dir)
        with open(os.path.join(out_dir, fileid + '.ann'), 'wt') as f:
            f.write('\t'.join(spantokens))

def text_to_tokens(text):
    """textをtoken単位に分割したリストを返す。"""
    tokens = re.split("""( | |\xa0|\t|\n|…|\'|\"|·|~|↔|•|\!|@|#|\$|%|\^|&|\*|
        -|=|_|\+|ˉ|\(|\)|\[|\]|\{|\}|;|‘|:|“|,|\.|\/|<|>|×|>|<|≤|≥|↑|↓|¬
        |®|•|′|°|~|≈|\?|Δ|÷|≠|‘|’|“|”|§|£|€|0|1|2|3|4|5|6|7|8|9|™|⋅)""", text)
    return list(filter(None, tokens))

def text_to_spantokens(text):
    """textをtokenizeし、(token, start_ix, end_ix)のリストとして返す。"""
    spantokens = []
    ix = 0
    for token in text_to_tokens(text):
        spantokens.append((token, ix, ix + len(token)))
        ix += len(token)
    return spantokens

def annotations_to_spantokens(annotations):
    spantokens = set()
    for annotation in annotations:
        if annotation:
            token = annotation.split('\t')[-1]
            start = int(annotation.split('\t')[1].split(' ')[1])
            end = int(annotation.split('\t')[1].split(' ')[-1])
            if token:
                spantokens.add((token, start, end))
    return spantokens

def text_to_labels(text, spantokens):
    text_spantokens = text_to_spantokens(text)
    labels = [O for i in range(len(text_spantokens))]
    ann_ix = 0
    ann_spans = sorted([(start, end) for _, start, end in spantokens], key=lambda x: x[0])
    for i, (entity, start, end) in enumerate(text_spantokens):
        if ann_ix == len(ann_spans):
            break
        # startがann_ixがさすendより過ぎた時にはann_ixをincrementする。
        if ann_spans[ann_ix][1] < start:
            ann_ix += 1
        elif start == ann_spans[ann_ix][0] and end == ann_spans[ann_ix][1]:
            labels[i] = S
            ann_ix += 1
        elif start == ann_spans[ann_ix][0] and end < ann_spans[ann_ix][1]:
            labels[i] = B
        elif end == ann_spans[ann_ix][1] and start > ann_spans[ann_ix][0]:
            labels[i] = E
            ann_ix += 1
        elif start > ann_spans[ann_ix][0] and end < ann_spans[ann_ix][1]:
            labels[i] = M
        else:
            pass
    return labels

def labels_to_spantokens(self, labels, text):
    spantokens = set()
    text_spantokens = text_to_spantokens(text)
    entity = ''
    ann_start = 0
    pre_label = ''
    for text_ann, label in zip(text_spantokens, labels):
        pre_now = (pre_label, label)
        # あり得るラベル列の組み合わせ
        if pre_now in [(B, M), (B, E), (M, E), (E, B), (E, S),
                       (S, B), (O, S), (O, B), (O, M), (O, E)]:
            if label == S:
                spantokens.add(text_ann)
            elif label == B:
                ann_start = text_ann[1]
                entity += text_ann[0]
            elif label == M:
                entity += text_ann[0]
            elif label == E:
                entity += text_ann[0]
                spantokens.add((entity, ann_start, text_ann[2]))
                entity = ''
            elif label == O:
                entity += text_ann[0]
        pre_label = label
    return spantokens

if __name__ == '__main__':
    for mode in ['train', 'test', 'valid']:
        make_chemdner_data(mode=mode)