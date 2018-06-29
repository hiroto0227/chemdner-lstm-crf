from scripts import datasets
from scripts import transformer
from scripts import trainer
from scripts import models
import config
from tqdm import tqdm



def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

if __name__ == '__main__':
    train_texts, train_anns = datasets.short()
    tf = transformer.WordLevelTransformer()
    tf.fit(''.join([text for t in train_texts for text in t]))
    embed_vecs = []
    label_vecs = []
    for text, anns in tqdm(zip(train_texts, train_anns)):
        embed_vecs.extend(tf.text2embedvec(text))
        labels = tf.text_anns2labels(text, anns)
        label_vecs.extend(tf.label_encoder.transform(labels))
        ############# test #################
        #print('############# test #################')
        #inv_annset = tf.labels_text2annset(labels, text)
        #[print(ann) for ann in anns]
        #print('----------')
        #[print(ann) for ann in inv_annset]
        ##################################
    print(numpy.array(embed_vecs).shape)
    print(numpy.array(label_vecs).shape)
    model = LSTMModel(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    
    for epoch in range(300):
        for sentence, tags in training_data:
            model.hidden = model.init_hidden()
            sentence_in = prepare_sequence(sentence, word_to_ix)
            targets = prepare_sequence(tags, tag_to_ix)
            tag_scores = model(sentence_in)
            loss = loss_function(tag_scores, targets)
            loss.backward()
            optimizer.step()
