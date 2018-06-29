import re
import numpy as np
from sklearn.preprocessing import LabelEncoder


def annfile2annset(filepath):
    """*.annからannの集合を返す。
    ann: (entity, start, end)
    """
    ann_set = set()
    with open(filepath, 'rt') as f:
        annotations = f.read().split('\n')
    for annotation in annotations:
        if annotation:
            entity = annotation.split('\t')[-1]
            start = int(annotation.split('\t')[1].split(' ')[1])
            end = int(annotation.split('\t')[1].split(' ')[-1])
            if entity:
                ann_set.add((entity, start, end))
    return ann_set


class WordLevelTransformer:
    def __init__(self):
        self.token2id = {}
        self.id2token = []
        self.REGEX_TOKENIZE = re.compile('( | |\xa0|\t|\n|…|\'|\"|·|~|↔|•|\!|@|#|\$|%|\^|&|\*|-|=|_|\+|ˉ|\(|\)|\[|\]|\{|\}|;|‘|:|“|,|\.|\/|<|>|×|>|<|≤|≥|↑|↓|¬|®|•|′|°|~|≈|\?|Δ|÷|≠|‘|’|“|”|§|£|€|0|1|2|3|4|5|6|7|8|9|™|⋅)')
        self.token_encoder = LabelEncoder()
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(['B', 'M', 'E', 'S', 'O'])

    def text2tokens(self, text):
        return self.REGEX_TOKENIZE.split(text)

    def text2embedvec(self, text):
        tokens = self.text2tokens(text)
        return self.token_encoder.transform(tokens)

    def fit(self, text):
        self.token_encoder.fit(self.text2tokens(text))
        return True

    def text2anns(self, text):
        anns = []
        ix = 0
        for token in self.text2tokens(text):
            anns.append((token, ix, ix + len(token)))
            ix += len(token)
        return anns

    def text_anns2labels(self, text, anns):
        """ann_setとtextからlabel列を返す。
        anns : [(entity, start, end), (), (), ...()]
        """
        text_anns = self.text2anns(text)
        labels = ['O' for i in range(len(text_anns))]
        ann_ix = 0
        ann_spans = sorted([(start, end) for _, start, end in anns], key=lambda x: x[0])
        for i, (entity, start, end) in enumerate(text_anns):
            if ann_ix == len(ann_spans):
                break
            # startがann_ixがさすendより過ぎた時にはann_ixをincrementする。
            if ann_spans[ann_ix][1] < start:
                ann_ix += 1
            elif start == ann_spans[ann_ix][0] and end == ann_spans[ann_ix][1]:
                labels[i] = 'S'
                ann_ix += 1
            elif start == ann_spans[ann_ix][0] and end < ann_spans[ann_ix][1]:
                labels[i] = 'B'
            elif end == ann_spans[ann_ix][1] and start > ann_spans[ann_ix][0]:
                labels[i] = 'E'
                ann_ix += 1
            elif start > ann_spans[ann_ix][0] and end < ann_spans[ann_ix][1]:
                labels[i] = 'M'
            else:
                pass
        return labels

    def labels_text2annset(self, labels, text):
        ann_set = set()
        text_anns = self.text2anns(text)
        entity = ''
        ann_start = 0
        pre_label = ''
        for text_ann, label in zip(text_anns, labels):
            pre_now = (pre_label, label)
            # あり得るラベル列の組み合わせ
            if pre_now in [('B', 'M'), ('B', 'E'), ('M', 'E'), ('E', 'B'), ('E', 'S'), ('S', 'B'), ('O', 'S'), ('O', 'B'), ('O', 'M'), ('O', 'E')]:
                if label == 'S':
                    ann_set.add(text_ann)
                elif label == 'B':
                    ann_start = text_ann[1]
                    entity += text_ann[0]
                elif label == 'M':
                    entity += text_ann[0]
                elif label == 'E':
                    entity += text_ann[0]
                    ann_set.add((entity, ann_start, text_ann[2]))
                    entity = ''
                elif label == 'O':
                    entity += text_ann[0]
            pre_label = label
        return ann_set
