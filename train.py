from crf import load_data, make_labels2i
from typing import List
import spacy
import torch
from crf import f1_score, predict, PAD_SYMBOL, pad_features, pad_labels
from tqdm.autonotebook import tqdm
import random
from crf import build_features_set
from crf import make_features_dict
from crf import encode_features, encode_labels
from crf import NERTagger

train_filepath = "./NER_TRAIN_JUDGEMENT.json"
dev_filepath = "./NER_DEV_JUDGEMENT.json"
train_sents, train_tag_sents = load_data(train_filepath)
dev_sents, dev_tag_sents = load_data(dev_filepath)
labels2i = make_labels2i()

print("train sample", train_sents[2], train_tag_sents[2])
print()
print("labels2i", labels2i)


def make_features(text: List[str], sent_tags: List[str]) -> List[List[int]]:
    feature_lists = []
    for i, token in enumerate(text):
        feats = []
        # We add a feature for each unigram.
        if i > 0:
          prev_word = text[i-1]
          prev_pos  = sent_tags[i-1]
        else:
          prev_word = '<s>'
          prev_pos  = "<s>"
        if i < len(text)-1:
          next_word = text[i+1]
          next_pos  = sent_tags[i+1]
        else:
          next_word = '<s>'
          next_pos  = '<s>'
        feats.append(f"word={token}")
        feats.append(f"pos={sent_tags[i]}")
        feats.append(f"prev_word={prev_word}")
        feats.append(f"prev_pos={prev_pos}")
        feats.append(f"next_word={next_word}")
        feats.append(f"next_pos={next_pos}")
        feature_lists.append(feats)
    return feature_lists

def featurize(sents: List[List[str]]) -> List[List[List[str]]]:
    nlp = spacy.load("en_core_web_sm")
    feats = []
    for sent in sents:
        sent_tags = []
        docs = [nlp(word) for word in sent]
        for doc in docs:
          for token in doc:
            sent_tags.append(token.pos_)
        if(len(sent_tags) == len(sent)):
          feats.append(make_features(sent, sent_tags))

    return feats

def training_loop(
    num_epochs,
    batch_size,
    train_features,
    train_labels,
    dev_features,
    dev_labels,
    optimizer,
    model,
    labels2i,
    pad_feature_idx
):
    samples = list(zip(train_features, train_labels))
    random.shuffle(samples)
    batches = []
    for i in range(0, len(samples), batch_size):
        batches.append(samples[i:i+batch_size])
    
    print("Training...")
    for i in range(num_epochs):
        losses = []
        for batch in tqdm(batches):
            features, labels = zip(*batch)
            features = pad_features(features, pad_feature_idx)
            features = torch.stack(features)
            labels = pad_labels(labels, labels2i[PAD_SYMBOL])
            labels = torch.stack(labels)
            mask = (labels != labels2i[PAD_SYMBOL])
            optimizer.zero_grad()
            log_likelihood = model(features, labels, mask=mask)
            negative_log_likelihood = -log_likelihood
            negative_log_likelihood.backward()
            optimizer.step()
            losses.append(negative_log_likelihood.item())
        
        print(f"epoch {i}, loss: {sum(losses)/len(losses)}")
        dev_f1 = f1_score(predict(model, dev_features), dev_labels, labels2i['O'])
        print(f"Dev F1 {dev_f1}")

    return model

train_features = featurize(train_sents[:500])
dev_features = featurize(dev_sents[:100])
all_features = build_features_set(train_features)
features_dict = make_features_dict(all_features)
model = NERTagger(len(features_dict), len(labels2i))

encoded_train_features = encode_features(train_features, features_dict)
encoded_dev_features = encode_features(dev_features, features_dict)
encoded_train_labels = encode_labels(train_tag_sents, labels2i)
encoded_dev_labels = encode_labels(dev_tag_sents, labels2i)
num_epochs = 1
batch_size = 16
LR=0.05
optimizer = torch.optim.SGD(model.parameters(), LR)

model = training_loop(
    num_epochs,
    batch_size,
    encoded_train_features,
    encoded_train_labels,
    encoded_dev_features,
    encoded_dev_labels,
    optimizer,
    model,
    labels2i,
    features_dict[PAD_SYMBOL]
)