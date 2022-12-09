import json
from spacy.training import offsets_to_biluo_tags
import spacy
from spacy.tokens import DocBin
from tqdm import tqdm

def preprocess_json(file_name):
    f = open(file_name)
    original = json.load(f)
    final = []
    for entry in original:
        text = entry['data']['text']
        entities = []
        annotations = entry['annotations'][0]['result']
        for annotation in annotations:
            entity = (annotation['value']['start'], annotation['value']['end'], annotation['value']['text'])
            entities.append(entity)
        final.append((text, {'entities': entities}))
    return final

def convert_json_spacy_format(preprocessed_train):
    nlp = spacy.blank("en")
    db = DocBin()
    for text, annot in tqdm(preprocessed_train):
        doc = nlp.make_doc(text) # create doc object from text
        ents = []
        for start, end, label in annot["entities"]: # add character indexes
            span = doc.char_span(start, end, label=label, alignment_mode="contract")
            if span is None:
                print("Skipping entity")
            else:
                ents.append(span)
        try:
            doc.ents = ents # label the text with the ents
            db.add(doc)
        except:
            print(text, annot)
    db.to_disk("./train.spacy")

def convert_spacy_format_iob(preprocessed_train):
    nlp = spacy.load('en_core_web_sm')
    docs = []
    for text, annot in preprocessed_train:
        doc = nlp(text)
        tags = offsets_to_biluo_tags(doc, annot['entities'])
        print(tags)

