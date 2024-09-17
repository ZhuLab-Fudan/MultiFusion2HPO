#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Split protein list into three parts: train set, ltr set, and test set.
We will get three annotation datasets, three protein lists, and term list.
Besides, we will split HPO terms into several groups according to frequency.
"""
import json
import random
from collections import defaultdict
from functools import reduce
from ontology import HumanPhenotypeOntology
from ontology import get_root, get_subontology


with open("../../config/preprocessing/split_dataset_temporal_22.json") as fp:
    config = json.load(fp)
  # "raw_annotation": {
  #   "time0": "../../data/annotation/clean/hpo_annotation_20191115.json",
  #   "time1": "../../data/annotation/clean/hpo_annotation_20201207.json",
  #   "time2": "../../data/annotation/clean/hpo_annotation_20210413.json",
  #   "time3": "../../data/annotation/clean/hpo_annotation_20220414.json",
  #   "time4": "../../data/annotation/clean/hpo_annotation_20230127.json",
  #   "time5": "../../data/annotation/clean/hpo_annotation_20230405.json",
  #   "time6": "../../data/annotation/clean/hpo_annotation_20230721.json",
  #   "time7": "../../data/annotation/clean/hpo_annotation_20230901.json",
  #   "time8": "../../data/annotation/clean/hpo_annotation_20231009.json",
  #   "time9": "../../data/annotation/clean/hpo_annotation_20240426.json"
  # },

# load various versions of HPO annotations
with open(config["raw_annotation"]["time0"]) as fp:
    annotation_t0 = json.load(fp)
with open(config["raw_annotation"]["time1"]) as fp:
    annotation_t1 = json.load(fp)
with open(config["raw_annotation"]["time2"]) as fp:
    annotation_t2 = json.load(fp)
with open(config["raw_annotation"]["time3"]) as fp:
    annotation_t3 = json.load(fp)
with open(config["raw_annotation"]["time4"]) as fp:
    annotation_t4 = json.load(fp)
with open(config["raw_annotation"]["time5"]) as fp:
    annotation_t5 = json.load(fp)
with open(config["raw_annotation"]["time6"]) as fp:
    annotation_t6 = json.load(fp)
with open(config["raw_annotation"]["time7"]) as fp:
    annotation_t7 = json.load(fp)
with open(config["raw_annotation"]["time8"]) as fp:
    annotation_t8 = json.load(fp)
with open(config["raw_annotation"]["time9"]) as fp:
    annotation_t9 = json.load(fp)

# load various versions of HPO
ontology = HumanPhenotypeOntology(config["ontology"]["path"],
                                     version=config["ontology"]["version"])

# split proteins in basic train set, ltr train set and test set
basic_protein = list(set(annotation_t2.keys()))
# random.seed(1)
# random.shuffle(basic_protein)
ltr_protein = list(set(annotation_t3.keys()) - set(annotation_t2.keys()))
test_protein = list(set(annotation_t9.keys()) - set(annotation_t3.keys()) - set(annotation_t2.keys()))

# HPO annotations for basic model training
basic_annotation = {p: annotation_t2[p] for p in basic_protein}

# HPO annotations for learning to rank
ltr_annotation = {p: annotation_t3[p] for p in ltr_protein}

test_annotation = {p: [] for p in test_protein}

# HPO annotations for evaluation

for protein in test_protein:
    print(protein)
    for hpo_term in annotation_t9[protein]:
        # print(hpo_term)
        # if "veteran" HPO term, just copy down
        if hpo_term in ontology:
            test_annotation[protein].append(hpo_term)
        # else if has old, alternative HPO terms, replace it
        elif hpo_term in ontology.alt_ids:
            for alternative in ontology.alt_ids[hpo_term]:
                print("Replace (%s, %s) to (%s, %s)" % (protein, hpo_term, protein, alternative))
                test_annotation[protein].append(alternative)
        # if not found, then discard
        else:
            print("Discard (%s, %s)" % (protein, hpo_term))

# output annotation
with open(config["processed_annotation"]["train"], 'w') as fp:
    json.dump(basic_annotation, fp, indent=2)
with open(config["processed_annotation"]["ltr"], 'w') as fp:
    json.dump(ltr_annotation, fp, indent=2)
with open(config["processed_annotation"]["test"], 'w') as fp:
    json.dump(test_annotation, fp, indent=2)

# output protein lists
with open(config["protein_list"]["train"], 'w') as fp:
    json.dump(list(basic_annotation.keys()), fp, indent=2)
with open(config["protein_list"]["ltr"], 'w') as fp:
    json.dump(list(ltr_annotation.keys()), fp, indent=2)
with open(config["protein_list"]["test"], 'w') as fp:
    json.dump(list(test_annotation.keys()), fp, indent=2)

train_annotation=basic_annotation
# merge three annotations into one dict
combined_annotation = {**train_annotation,
                       **ltr_annotation,
                       **test_annotation}

# propagate annotations and discard roots (All & sub-ontology)
# propagate annotations and discard roots (All & sub-ontology)
propagated_combined_annotation = dict()
for protein in combined_annotation:
    propagated_combined_annotation[protein] = list(
        ontology.transfer(combined_annotation[protein]) - {get_root()} -
        set(get_subontology(ontology.version)))

# write down used HPO terms into file
with open(config["term_list"], 'w') as fp:
    json.dump(list(reduce(lambda a, b: set(a) | set(b),
                          propagated_combined_annotation.values())),
              fp, indent=2)

# split HPO terms according to its frequency
# 1. count the frequency of HPO terms
#    term_counts = { hpo_term1: cnt1, hpo_term2: cnt2, ... }
term_counts = defaultdict(int)
for protein in propagated_combined_annotation:
    for term in propagated_combined_annotation[protein]:
        term_counts[term] += 1
# 2. write term list each interval
for interval in config["frequency"]:
    with open(interval["path"], 'w') as fp:
        json.dump([term for term in term_counts
                   if interval["low"] <= term_counts[term] <= interval["high"]],
                  fp, indent=2)
