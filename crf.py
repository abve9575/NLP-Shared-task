from typing import Dict, List, Set, Tuple

import torch
from torchcrf import CRF as CRFDecoder
from tqdm import tqdm
import random
import json
import spacy
from spacy.training import offsets_to_biluo_tags

UNK_SYMBOL = "<UNK>"
PAD_SYMBOL = "<PAD>"

def load_data(
    filepath: str,
) -> Tuple[List[str], List[str]]:

    labels2i = {'<PAD>': 0, 'B-PRECEDENT': 1, 'B-RESPONDENT': 2, 'B-COURT': 3, 'B-PETITIONER': 4, 'B-PROVISION': 5, 'B-LAWYER': 6, 'B-STATUTE': 7, 'B-CASE_NUMBER': 8, 'B-DATE': 9, 'B-OTHER_PERSON': 10, 'B-JUDGE': 11, 'B-ORG': 12, 'B-GPE': 13, 'B-WITNESS': 14, 'I-PRECEDENT': 15, 'I-RESPONDENT': 16, 'I-COURT': 17, 'I-PETITIONER': 18, 'I-PROVISION': 19, 'I-LAWYER': 20, 'I-STATUTE': 21, 'I-CASE_NUMBER': 22, 'I-DATE': 23, 'I-OTHER_PERSON': 24, 'I-JUDGE': 25, 'I-ORG': 26, 'I-GPE': 27, 'I-WITNESS': 28, 'O': 29}
    f = open(filepath)
    original = json.load(f)
    preprocessed_train = []
    for entry in original:
        text = entry['data']['text']
        entities = []
        annotations = entry['annotations'][0]['result']
        for annotation in annotations:
            entity = (annotation['value']['start'], annotation['value']['end'], annotation['value']['labels'][0])
            entities.append(entity)
        preprocessed_train.append((text, {'entities': entities}))
    
    nlp = spacy.load('en_core_web_sm')
    all_tags = []
    all_sentences = []
    for text, annot in preprocessed_train:
        doc = nlp(text)
        tags = offsets_to_biluo_tags(doc, annot['entities'])
        tags = [t.replace('L-', 'I-') for t in tags]
        tags = [t.replace('U-', 'B-') for t in tags]
        valid_tags = True
        for tag in tags:
            if tag not in labels2i.keys():
                print("invalid tag detected", tag, text)
                valid_tags = False
                break
        sent = [token.text for token in doc]
        if valid_tags:
            all_tags.append(tags)
            all_sentences.append(sent)
    return all_sentences, all_tags


def make_labels2i():
    return {'<PAD>': 0, 'B-PRECEDENT': 1, 'B-RESPONDENT': 2, 'B-COURT': 3, 'B-PETITIONER': 4, 'B-PROVISION': 5, 'B-LAWYER': 6, 'B-STATUTE': 7, 'B-CASE_NUMBER': 8, 'B-DATE': 9, 'B-OTHER_PERSON': 10, 'B-JUDGE': 11, 'B-ORG': 12, 'B-GPE': 13, 'B-WITNESS': 14, 'I-PRECEDENT': 15, 'I-RESPONDENT': 16, 'I-COURT': 17, 'I-PETITIONER': 18, 'I-PROVISION': 19, 'I-LAWYER': 20, 'I-STATUTE': 21, 'I-CASE_NUMBER': 22, 'I-DATE': 23, 'I-OTHER_PERSON': 24, 'I-JUDGE': 25, 'I-ORG': 26, 'I-GPE': 27, 'I-WITNESS': 28, 'O': 29}


    
class NERTagger(torch.nn.Module):
    def __init__(self, features_dim: int, num_tags: int):
        super().__init__()
        self.num_tags = num_tags
        self.features_dim = features_dim
        self.emissions_scorer = torch.nn.Embedding(features_dim, num_tags)
        self.crf_decoder = CRFDecoder(self.num_tags, batch_first=True)

    def forward(
        self, input_seq: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        emissions = self.make_emissions(input_seq)
        return self.crf_decoder(emissions, tags, mask=mask, reduction="mean")

    def decode(self, input_seq: torch.Tensor) -> List[int]:
        emissions = self.make_emissions(input_seq)
        # Decode the argmax sequence of labels with viterbi
        return self.crf_decoder.decode(emissions)

    def make_emissions(self, input_seq: torch.Tensor) -> torch.Tensor:
        return torch.sum(self.emissions_scorer(input_seq), dim=-2)


def precision(
    predicted_labels: List[torch.Tensor],
    true_labels: List[torch.Tensor],
    outside_tag_idx: int
):
    TP = torch.tensor([0])
    denom = torch.tensor([0])
    for pred, true in zip(predicted_labels, true_labels):
        TP += sum((pred == true)[pred != outside_tag_idx])
        denom += sum(pred != outside_tag_idx)

    # Avoid division by 0
    denom = torch.tensor(1) if denom == 0 else denom
    return TP / denom


def recall(
    predicted_labels: List[torch.Tensor],
    true_labels: List[torch.Tensor],
    outside_tag_idx: int
):
    TP = torch.tensor([0])
    denom = torch.tensor([0])
    for pred, true in zip(predicted_labels, true_labels):
        TP += sum((pred == true)[true != outside_tag_idx])
        denom += sum(true != outside_tag_idx)

    # Avoid division by 0
    denom = torch.tensor(1) if denom == 0 else denom
    return TP / denom


def f1_score(predicted_labels, true_labels, outside_tag_idx):
    P = precision(predicted_labels, true_labels, outside_tag_idx)
    R = recall(predicted_labels, true_labels, outside_tag_idx)
    return 2*P*R/(P+R)
        

def make_features_dict(all_features: Set) -> Dict:
    features_dict = {f: i+2 for i, f in enumerate(all_features)}
    features_dict[PAD_SYMBOL] = 0
    features_dict[UNK_SYMBOL] = 1
    print(f"Found {len(features_dict)} features")

    return features_dict


def encode_token_features(features: List[str], features_dict: Dict[str, int]) -> torch.Tensor:
    return torch.LongTensor([
        features_dict.get(feat, features_dict[UNK_SYMBOL]) for feat in features
    ])


def predict(model: torch.nn.Module, feature_sents: List[List[int]]) -> List[torch.Tensor]:
    out = []
    for features in feature_sents:
        # Dummy batch dimension
        features = features.unsqueeze(0)
        # -> List[List[int]]
        preds = model.decode(features)
        preds = [torch.tensor(p) for p in preds]
        out.extend(preds)

    return out

def pad_tensor(tensor: torch.Tensor, pad_max: int, pad_idx: int) -> torch.Tensor:
    padding = pad_max - len(tensor)
    return torch.nn.functional.pad(tensor, (0, padding), "constant", pad_idx)


def pad_2d_tensor(
    tensor: torch.Tensor,
    pad_max: int,
    num_features: int,
    pad_idx: int
) -> torch.Tensor:
    padding_len = pad_max - len(tensor)
    pads_matrix = torch.ones(padding_len, num_features) * pad_idx
    return torch.cat((tensor, pads_matrix.long()))


def pad_labels(labels_list: List[torch.Tensor], pad_idx: int) -> List[torch.Tensor]:
    pad_max = max([len(l) for l in labels_list])
    return [pad_tensor(l, pad_max, pad_idx) for l in labels_list]


def pad_features(features_list: List[torch.Tensor], pad_idx: int) -> List[torch.Tensor]:
    pad_max = max([len(l) for l in features_list])
    num_features = features_list[0].size(1)
    return [pad_2d_tensor(f, pad_max, num_features, pad_idx) for f in features_list]


def build_features_set(train_features: List[List[List[str]]]) -> Set:
    all_features = set()

    print("Building features set!")
    for features_sent in tqdm(train_features):
        for features in features_sent:
            for f in features:
                all_features.add(f)

    return all_features


def encode_features(
    feature_sents: List[List[List[str]]], features_dict: Dict[str, int]
) -> List[torch.LongTensor]:
    encoded_features = []
    
    for feature_sent in feature_sents:
        encoded = [
            encode_token_features(token_features, features_dict) for token_features in feature_sent
        ]
        encoded_features.append(torch.stack(encoded))

    return encoded_features


def encode_labels(tag_sents: List[List[str]], labels2i: Dict[str, int]) -> List[torch.LongTensor]:
    encoded_labels = []
    for labels in tag_sents:
        encoded_labels.append(torch.LongTensor([
            labels2i[l] for l in labels
        ]))

    return encoded_labels