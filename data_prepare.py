import os, sys, re, csv
import utils
from utils import S, B, M, E, O, NEWLINE, UNK, PAD, EOS, SOS, COMMA


def annfile_to_annset(annotations):
    ann_set = set()
    for annotation in annotations:
        if annotation:
            entity = annotation.split('\t')[-1]
            start = int(annotation.split('\t')[1].split(' ')[1])
            end = int(annotation.split('\t')[1].split(' ')[-1])
            if entity:
                ann_set.add((entity, start, end))
    return ann_set


def to_tokenlabel_csv(outfilepath, tokens, labels):
    with open(outfilepath, 'wt') as f:
        csv_writer = csv.writer(f)
        for token, label in zip(tokens, labels):
            token = token if token != ',' else utils.COMMA
            token = token if token != '\n' else utils.NEWLINE
            csv_writer.writerow([token, label])


if __name__ == '__main__':
    all_tokens = {COMMA, NEWLINE, UNK, PAD, SOS, EOS}
    all_labels = {S, B, M, E, O, PAD, SOS, EOS}
    in_dir_path = sys.argv[1]
    out_dir_path = sys.argv[2]
    fileids = [f[:-4] for f in os.listdir(in_dir_path) if f.endswith('.txt')]
    for fileid in fileids:
        with open('{}/{}.txt'.format(in_dir_path, fileid), 'rt') as f:
            text = f.read()
        with open('{}/{}.ann'.format(in_dir_path, fileid), 'rt') as f:
            annotations = f.read().split('\n')
            anns = annfile_to_annset(annotations)
        tokens = utils.text_to_tokens(text)
        labels = utils.text_anns_to_labels(text, anns)
        all_tokens = all_tokens.union(set(tokens))
        all_labels = all_labels.union(set(labels))
        to_tokenlabel_csv('{}/{}_tokenlabel.csv'.format(out_dir_path, fileid), tokens, labels)
    # save vacab and tags
    with open(out_dir_path + 'token_to_id', 'wt') as f:
        for token in sorted(all_tokens):
            token = token if token != ',' else utils.COMMA
            f.write('{}\n'.format(token))
    with open(out_dir_path + 'label_to_id', 'wt') as f:
        for label in sorted(all_labels):
            f.write('{}\n'.format(label))