from scripts import datasets
from scripts import transformer
from scripts import models
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


if __name__ == '__main__':
    train_texts, train_anns = datasets.short()
    tf = transformer.WordLevelTransformer()
    tf.fit(''.join([text for t in train_texts for text in t]))
    token_ids = []
    label_ids = []
    for text, anns in tqdm(zip(train_texts, train_anns)):
        token_ids.append(tf.text2ids(text))
        labels = tf.text_anns2labels(text, anns)
        label_ids.append(tf.label_encoder.transform(labels))
        ############# test #################
        #print('############# test #################')
        #inv_annset = tf.labels_text2annset(labels, text)
        #[print(ann) for ann in anns]
        #print('----------')
        #[print(ann) for ann in inv_annset]
        ##################################
    model = models.LSTM_CRF(len(tf.token_encoder.classes_), len(tf.label_encoder.classes_))
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    for epoch in tqdm(range(3)):
        for i, (token_id, label_id) in enumerate(zip(token_ids, label_ids)):
            try:
                inputs = torch.tensor(token_id, dtype=torch.long)
                targets = torch.tensor(label_id, dtype=torch.long)
                loss = torch.mean(model(inputs, targets))
                loss.backward()
                optimizer.step()
                print('--------{}: {}'.format(i, loss))
            except KeyboardInterrupt:
                tf.save('outsource/tf')
                torch.save(model, 'outsource/mymodel.pth')
    # save
    tf.save('outsource/tf')
    torch.save(model, 'outsource/mymodel.pth')
