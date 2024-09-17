import os
import json
import pickle


import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel


with open("../../../config/feature/else/BioLinkBERT/biolinkbert.json") as fp:
    config = json.load(fp)

with open(config["abstract"]) as fp:
    abstracts = json.load(fp)

tokenizer = AutoTokenizer.from_pretrained('/public/home/zhaiwq/modelused/DeepText2HPO_pycharm_project/src/feature/else/BioLinkBERT-large')
model = AutoModel.from_pretrained('/public/home/zhaiwq/modelused/DeepText2HPO_pycharm_project/src/feature/else/BioLinkBERT-large')#.cuda()
feature = {}
# for protein in abstracts:
#     texts = abstracts[protein]
#
#
#     inputs = tokenizer(texts, return_tensors="pt")
#
#     outputs = model(**inputs)
#     last_hidden_states = outputs.last_hidden_state
#
#     last_hidden_states_pool_output = outputs.pooler_output
#     feature_vectors = last_hidden_states_pool_output.tolist()[0]
#
#
#
#     feature.update({protein: feature_vectors})

for protein in abstracts:
    texts = abstracts[protein]

    # replace with your own list of entity names
    # all_names = ["covid-19", "Coronavirus infection", "high fever", "Tumor of posterior wall of oropharynx"]

    all_names=[]
    all_names.append(texts)
    bs = 2048 # batch size during inference
    all_embs = []
    for i in tqdm(np.arange(0, len(all_names), bs)):
        toks = tokenizer.batch_encode_plus(all_names[i:i+bs],
                                           padding="max_length",
                                           max_length=25,
                                           truncation=True,
                                           return_tensors="pt")
        toks_cuda = {}
        for k,v in toks.items():
            # toks_cuda[k] = v.cuda()
            toks_cuda[k] = v#.cuda()
        cls_rep = model(**toks_cuda)[0][:,0,:] # use CLS representation as the embedding
        all_embs.append(cls_rep.cpu().detach().numpy())

    all_embs = np.concatenate(all_embs, axis=0)


    feature_vectors= np.array(all_embs).tolist()[0]
    feature.update({protein: feature_vectors})


with open(config["output"], 'w') as fp:
    json.dump(feature, fp, indent=2)