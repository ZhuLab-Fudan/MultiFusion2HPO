import gensim
from gensim.models import Doc2Vec
import json

with open("../../../config/feature/D2V/d2v.json") as fp:
    config = json.load(fp)

with open(config["abstract"]) as fp:
    abstracts = json.load(fp)

documents = []
for protein in abstracts:
    words = abstracts[protein].split()
    documents.append(gensim.models.doc2vec.TaggedDocument(words, [protein]))

model_dm = Doc2Vec(dm=1, min_count=1, window=10, vector_size=3000, epochs=30, workers=10)
model_dbow = Doc2Vec(dm=0, min_count=1, window=10, vector_size=3000, epochs=30, workers=10)
# model_dbow = Doc2Vec(dm=0, min_count=3, window=8, vector_size=200, epochs=30, workers=10, negative=5)
model_dm.build_vocab(documents)
model_dbow.build_vocab(documents)

model_dm.train(documents, total_examples=model_dm.corpus_count, epochs=model_dm.epochs)
model_dbow.train(documents, total_examples=model_dbow.corpus_count, epochs=model_dbow.epochs)

feature = {}
for protein in abstracts:
    feature_dm = model_dm.docvecs[protein].tolist()
    feature_dbow = model_dbow.docvecs[protein].tolist()
    feature.update({protein: feature_dbow})

with open(config["output"], 'w') as fp:
    json.dump(feature, fp, indent=2)