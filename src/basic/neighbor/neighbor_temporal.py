#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Implementation of Neighbor method.

It propagates HPO terms from neighbors in PPI network.
"""
import sys
import json
import time
from collections import defaultdict
from functools import reduce
from multiprocessing import Pool
from ontology import HumanPhenotypeOntology
from file_reader import load_protein, load_annotation


def _predict_job(network, annotation, protein_list):
    """Sub-routine to calculate predictive scores.
    :param network: protein-protein interaction network
    :param annotation: HPO annotations of training set
    :param protein_list: list of proteins assigned to this sub-routine
    :return: predictive score, like
        { protein1: { hpo_term1: score1, ... }, ... }
    """
    scores = defaultdict(dict)
    for protein in protein_list:
        if protein in network:
            hpo_terms = reduce(lambda a, b: a | b,
                               [set(annotation.get(neighbour, set()))
                                for neighbour in network[protein]])
            normalizer = sum(network[protein].values())
            for hpo_term in hpo_terms:
                scores[protein][hpo_term] = sum(
                    [(hpo_term in annotation.get(neighbour, set())) *
                     network[protein][neighbour]
                     for neighbour in network[protein]]) / normalizer
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
              "Predict", protein)
    return scores


def neighbor_scoring(network, train_annotation, test_proteins, num_worker=1):
    """Scoring function of Neighbor method.
    :param network: protein-protein interaction network, like
        { protein1: { protein_a: score1a, ... }, ... }
    :param train_annotation: HPO annotations of training set
        { protein1: [ hpo_term1, hpo_term2, ... ], ... }
    :param test_proteins: list of proteins in test set
        [ protein1, protein2, ... ]
    :return: predictive score, like
        { protein1: { hpo_term1: score1, ... }, ... }
    """
    if num_worker > 1:
        res_ids = list()
        # start parallel computing
        pool = Pool(num_worker)
        for i in range(num_worker):
            sub_protein_list = test_proteins[i::num_worker]
            sub_network = {protein: network[protein]
                           for protein in sub_protein_list
                           if protein in network}
            res_id = pool.apply_async(
                _predict_job,
                args=(sub_network, train_annotation, sub_protein_list)
            )
            res_ids.append(res_id)
        pool.close()
        pool.join()
        # achieve final results
        scores = reduce(lambda a, b: {**a, **b},
                        [res_id.get() for res_id in res_ids])
    else:
        scores = _predict_job(network, train_annotation, test_proteins)
    return scores


def add_weight(network):
    """Add weight to the edges in the network.
    :param network: PPI network (but scores are 0/1)
        { protein1: { protein1a: score1a, protein1b: score1b, ... },
          protein2: { protein2a: score2a, protein2b: score2b, ... },
          ... }
    :return: weighted PPI network, dict
        { protein1: { protein1a: score1a, protein1b: score1b, ... },
          protein2: { protein2a: score2a, protein2b: score2b, ... },
          ... }
    """
    ppi = defaultdict(dict)
    for protein_a in network:
        neighbour_a = set(network[protein_a].keys()) | {protein_a}
        for protein_b in network[protein_a]:
            neighbour_b = set(network[protein_b].keys()) | {protein_b}
            score = ((2 * len(neighbour_a & neighbour_b)) /
                     (len(neighbour_a - neighbour_b) +
                      2 * len(neighbour_a & neighbour_b) + 1)) * \
                    ((2 * len(neighbour_a & neighbour_b)) /
                     (len(neighbour_b - neighbour_a) +
                      2 * len(neighbour_a & neighbour_b) + 1))
            ppi[protein_a][protein_b] = score
            ppi[protein_b][protein_a] = score
    return ppi


if __name__ == "__main__":
    conf_path = "../../../config/basic/neighbor/neighbor_STRING_temporal_22.json"
    if len(sys.argv) > 1 and len(sys.argv[1]) > 0:
        conf_path = sys.argv[1]

    with open(conf_path) as fp:
        config = json.load(fp)

    # load PPI network
    with open(config["network"]["path"]) as fp:
        network = json.load(fp)
    # actually customized for BioGRID
    if config["network"]["type"] == "unweighted":
        network = add_weight(network)

    # load proteins in training set and test set
    ltr_proteins = load_protein(config["protein_list"]["ltr"])
    test_proteins = load_protein(config["protein_list"]["test"])

    # load HPO
    ontology = HumanPhenotypeOntology(config["ontology"]["path"],
                                      version=config["ontology"]["version"])
    # load HPO annotations of training set
    train_annotation = load_annotation(config["annotation"], ontology, ns="all")

    # scoring by Neighbor method
    ltr_result = neighbor_scoring(network, train_annotation, ltr_proteins)
    test_result = neighbor_scoring(network, train_annotation, test_proteins)

    # write into file
    with open(config["result"]["ltr"], 'w') as fp:
        json.dump(ltr_result, fp)#, orient='index')# indent=2, orient='index')
    with open(config["result"]["test"], 'w') as fp:
        json.dump(test_result, fp)#, orient='index')# indent=2, orient='index')
