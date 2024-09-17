#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create HPO annotations without propagation from raw file.

Output format:
{ protein_id1: [ hpo_term1, hpo_term2, ... ],
  protein_id2: [ hpo_term1, hpo_term2, ... ],
  ...
}
"""
import json
from collections import defaultdict
from file_reader import gene2uniprot
from ontology import HumanPhenotypeOntology

with open("../../config/preprocessing/create_annotation.json") as fp:
    config = json.load(fp)

# load mapping of gene id to uniprot id
gene2protein = gene2uniprot(config["mapping"], gene_column=0, uniprot_column=1)

# load hpo annotations without propagation
annotation = defaultdict(list)
ontology_t0 = HumanPhenotypeOntology(config["ontology"]["path"], version=config["ontology"]["version"])
# terms = list(ontology_t0.keys())
with open(config["raw_annotation"]) as fp:
    for line in fp:
        if line.startswith('n'):
            continue
        if line.startswith('#'):
            continue
        # #use in annotation 20191115
        # gene_id, _, _, hpo_term = line.strip().split('\t')
        #use in annotation 202006
        gene_id = line.strip().split('\t')[0]
        # print(gene_id)
        hpo_term = line.strip().split('\t')[2]
        #only use to extract new added hpo_term than hp_20190906
        # if hpo_term not in ontology_t0 and hpo_term not in ontology_t0.alt_ids:
        #     continue
        for protein_id in gene2protein[gene_id]:
            if hpo_term not in annotation[protein_id]:
                annotation[protein_id].append(hpo_term)

# output annotation
with open(config["processed_annotation"], 'w') as fp:
    json.dump(annotation, fp, indent=2)
