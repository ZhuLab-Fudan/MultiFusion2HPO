#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Implementation of Learning to Rank.

We adopt LambdaMART, one of state-of-the-art Learning to Rank technique,
to integrate different basic prediction results and further improve the
performance.
"""
import json
from collections import defaultdict
import math
import xgboost as xgb
import numpy as np
from ontology import HumanPhenotypeOntology, get_ns_id
from file_reader import load_protein, load_annotation, \
    load_result, load_label_list


# CONSTANT for missing value
MISSING = -999.0

class LTR:
    """With the help of XGBoost, we implement LambdaMART here.
    """
    def __init__(self):
        self.model = None

    def fit(self, res_list, protein_list, annotation, top, param, **kwargs):
        """Train the ranking system with the scores being features
        :param res_list: a list containing different predictive scores
            obtained from basic models, each entry is a dict like
            {protein1: {term1: score11, term2: score12, ...}, ...}
        :param protein_list: list of training proteins
        :param annotation: dict, containing HPO annotations as label, like
            {protein1: {term1, term2, ...}, protein2: {term3, ...}, ...}
        :param top: int, the number of candidates to be selected
        :param param: dict, hyper-parameters for setting up LambdaMART
        :param kwargs: settings for fitting model
        :return: None
        """
        # select top candidates for each prediction result and each protein
        candidate = defaultdict(set)
        for scores in res_list:
            for protein in scores:
                if protein not in protein_list or protein not in annotation:
                    continue
                for hpo_term, _ in sorted(scores[protein].items(),
                                          key=lambda x: x[1],
                                          reverse=True)[:top]:
                    candidate[protein].add(hpo_term)

        # prepare features (predictive scores obtained by basic models) and
        # labels (whether protein in candidate is annotated by hpo_term)
        features, labels = list(), list()
        for protein in candidate:
            for hpo_term in candidate[protein]:
                instance = list()
                for scores in res_list:
                    # set missing value as MISSING instead of zero
                    instance.append(scores.get(protein, {}).get(hpo_term, MISSING))
                features.append(instance)
                labels.append(int(hpo_term in annotation[protein]))

        # start training phase here
        features = np.array(features)
        labels = np.array(labels)
        dtrain = xgb.DMatrix(features, label=labels, missing=MISSING)
        model = xgb.train(param, dtrain,
                          evals=[(dtrain, 'train')], **kwargs)
        self.model = model

    def predict(self, res_list, protein_list, term_list):
        """Predict associated score between proteins in protein_list and HPO
           terms in term_list with the scores in res_list as features.
        :param res_list: a list containing different predictive scores
            obtained from basic models, each entry is a dict like
            {protein1: {term1: score11, term2: score12, ...}, ...}
        :param protein_list: list of test proteins
        :param term_list: list of HPO terms
        :return: dict, prediction results like
            {protein1: {term1: score11, term2: score12, ...}, ...}
        """
        # prepare features (predictive scores obtained by basic models) and
        # labels (whether protein in candidate is annotated by hpo_term)
        identifier, features = list(), list()
        for protein in protein_list:
            for hpo_term in term_list:
                identifier.append([protein, hpo_term])
                instance = list()
                for scores in res_list:
                    # set missing value as MISSING instead of zero
                    instance.append(scores.get(protein, {}).get(hpo_term, MISSING))
                features.append(instance)
        features = np.array(features)

        # make prediction here
        prediction = defaultdict(dict)
        if features.shape[0] == 0:
            return prediction
        dtest = xgb.DMatrix(features, missing=MISSING)
        for i, score in enumerate(self.model.predict(dtest)):
            protein, hpo_term = identifier[i][0], identifier[i][1]
            # map score to the floating number between 0 and 1
            score = 1 / (1 + math.exp(-score))
            prediction[protein][hpo_term] = score

        return prediction



if __name__ == "__main__":
    with open("../../../config/ensemble/ltr/ltr_with_more_all_full_esm2_geneexpression_biolinkbert_temporal_22_tfidfd2v.json") as fp:
        config = json.load(fp)

    # load HPO
    ontology = HumanPhenotypeOntology(config["ontology"]["path"],
                                      version=config["ontology"]["version"])
    # load list of namespace id
    ns_id = get_ns_id(version=config["ontology"]["version"])

    # load protein list of training & test test
    train_protein_list = load_protein(config["protein_list"]["ltr"])
    test_protein_list = load_protein(config["protein_list"]["test"])

    # load HPO term list
    term_list = load_label_list(config["term_list"])

    # Booster parameters
    model_param = config["model_param"]
    # parameters for xgboost.train()
    fit_param = config["fit_param"]
    # top-k for selecting scores
    top = config["top"]

    # load results of different basic models
    # NOTE: use list rather than set/dict to maintain the order
    train_result = list()
    for res_path in config["result"]["ltr"]:
        res = load_result(res_path)
        train_result.append(res)
    test_result = list()
    for res_path in config["result"]["test"]:
        res = load_result(res_path)
        test_result.append(res)

    # train & test by sub-ontology
    combined_pred = defaultdict(dict)
    combined_valid_pred = defaultdict(dict)
    for ns in ns_id:
        if ns in ["freq", "bg", "pmh"]:
            continue
        print("*** Start", ns)

        # load propagated HPO annotations of specified sub-ontology
        train_hpo_annotation_ns = load_annotation(config["annotation"]["ltr"],
                                                  ontology, ns=ns)
        test_hpo_annotation_ns = load_annotation(config["annotation"]["test"],
                                                 ontology, ns=ns)
        # select proteins used for training
        train_protein_ns = list(set(train_protein_list) &
                                set(train_hpo_annotation_ns.keys()))
        test_protein_ns = list(set(test_protein_list) &
                               set(test_hpo_annotation_ns.keys()))

        # filter out prediction only for HPO terms in this sub-ontology
        train_result_ns = list()
        for res in train_result:
            filtered_res = dict()
            # print(res)
            for protein in res:
                filtered = {k: v for k, v in res[protein].items()
                            if ontology[k].ns == ns}
                if len(filtered) > 0:
                    filtered_res[protein] = filtered
            train_result_ns.append(filtered_res)
        test_result_ns = list()
        for res in test_result:
            filtered_res = dict()
            for protein in res:
                filtered = {k: v for k, v in res[protein].items()
                            if ontology[k].ns == ns}
                if len(filtered) > 0:
                    filtered_res[protein] = filtered
            test_result_ns.append(filtered_res)
        # train_result_ns=train_result
        # test_result_ns=test_result
        # filter out HPO terms of this sub-ontology
        term_list_ns = [t for t in term_list if ontology[t].ns == ns]

        # train the model and use it to perform prediction
        ltr_model_ns = LTR()
        ltr_model_ns.fit(train_result_ns, train_protein_ns,
                         train_hpo_annotation_ns, top[ns],
                         model_param, **fit_param)
        pred_ns = ltr_model_ns.predict(test_result_ns,
                                       test_protein_ns,
                                       term_list_ns)

        pred_valid_ns = ltr_model_ns.predict(train_result_ns,
                                       train_protein_ns,
                                       term_list_ns)

        # combine result into final result
        for protein in pred_ns:
            for term in pred_ns[protein]:
                combined_pred[protein][term] = pred_ns[protein][term]

        for protein in pred_valid_ns:
            for term in pred_valid_ns[protein]:
                combined_valid_pred[protein][term] = pred_valid_ns[protein][term]

    # write result
    with open(config["prediction"], 'w') as fp:
        json.dump(combined_pred, fp)#, orient='index')# indent=2, orient='index')

    with open(config["valid_prediction"], 'w') as fp:
        json.dump(combined_valid_pred, fp)#, orient='index')# indent=2, orient='index')
