import re

S = 'S'
B = 'B'
M = 'M'
E = 'E'
O = 'O'
PAD = '<PAD>'
SOS = '<SOS>'
EOS = '<EOS>'
UNK = '<UNK>'
COMMA = '<COMMA>'
NEWLINE = '<NEWLINE>'


def text_to_anns(text):
    anns = []
    ix = 0
    for token in text_to_tokens(text):
        anns.append((token, ix, ix + len(token)))
        ix += len(token)
    return anns


def text_to_tokens(text):
    tokens = re.split("""( | |\xa0|\t|\n|…|\'|\"|·|~|↔|•|\!|@|#|\$|%|\^|&|\*|
        -|=|_|\+|ˉ|\(|\)|\[|\]|\{|\}|;|‘|:|“|,|\.|\/|<|>|×|>|<|≤|≥|↑|↓|¬
        |®|•|′|°|~|≈|\?|Δ|÷|≠|‘|’|“|”|§|£|€|0|1|2|3|4|5|6|7|8|9|™|⋅)""", text)
    return list(filter(None, tokens))


def text_anns_to_labels(text, anns):
    text_anns = text_to_anns(text)
    labels = [O for i in range(len(text_anns))]
    ann_ix = 0
    ann_spans = sorted([(start, end) for _, start, end in anns], key=lambda x: x[0])
    for i, (entity, start, end) in enumerate(text_anns):
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


def labels_text_to_anns(self, labels, text):
    anns = set()
    text_anns = text_to_anns(text)
    entity = ''
    ann_start = 0
    pre_label = ''
    for text_ann, label in zip(text_anns, labels):
        pre_now = (pre_label, label)
        # あり得るラベル列の組み合わせ
        if pre_now in [(B, M), (B, E), (M, E), (E, B), (E, S),
                       (S, B), (O, S), (O, B), (O, M), (O, E)]:
            if label == S:
                anns.add(text_ann)
            elif label == B:
                ann_start = text_ann[1]
                entity += text_ann[0]
            elif label == M:
                entity += text_ann[0]
            elif label == E:
                entity += text_ann[0]
                anns.add((entity, ann_start, text_ann[2]))
                entity = ''
            elif label == O:
                entity += text_ann[0]
        pre_label = label
    return anns

