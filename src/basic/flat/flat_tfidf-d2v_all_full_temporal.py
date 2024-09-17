#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Flat classification trained and tested on each HPO term.
"""
import sys
import json
import time
from collections import defaultdict
from functools import reduce
from multiprocessing import Pool
# import sharedmem
import pandas as pd
# import torch
from scipy import sparse
import numpy as np
from sklearn.linear_model import LogisticRegression
# from torch import nn
# from torch.utils.data import DataLoader
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from ontology import HumanPhenotypeOntology, get_ns_id
from file_reader import load_protein, load_annotation, load_feature

# class AutoEncoder(nn.Module):
#     def __init__(self):
#         super(AutoEncoder,self).__init__()
#         global feature_size
#         self.encoder  =  nn.Sequential(
#             nn.Linear(feature_size,1024),
#             nn.Tanh(),
#             nn.Linear(1024, 200)
#         )
#         self.decoder = nn.Sequential(
#             nn.Linear(200, 1024),
#             nn.Tanh(),
#             nn.Linear(1024, feature_size),
#             nn.Sigmoid()
#         )
#     def forward(self, x):
#         encoded = self.encoder(x)
#         decoded = self.decoder(encoded)
#         return encoded, decoded

def array_to_csr(array):
    """Convert Numpy array to SciPy sparse matrix.
    :param array: a Numpy array
    :return: if the density of matrix is less than 0.7, then return the
        contents of the matrix as a sparse SciPy CSR matrix, else return
        the input dense matrix
    """
    if array.size > 0 and np.count_nonzero(array) / array.size < 0.7:
        return sparse.csr_matrix(array)
    else:
        return array


class SameModel:
    """Works when only one class in ground truth.
    """
    def __init__(self):
        self.value = 0

    def fit(self, X, y):
        """Fill predictive score with the label (0/1) in label vector y.
        :param X: no use here
        :param y: label vector, can be list, numpy array or pd.DataFrame
        :return: None
        """
        if isinstance(y, pd.DataFrame):
            y = np.asarray(y)[:, 0]
        self.value = y[0]

    def predict(self, X):
        """Predict score using the label in label vector (0/1).
        :param X: feature matrix, its size of rows is useful here.
        :return: a numpy matrix of shape (n_samples, 2), the first column is
            false label's (i.e. 0), the second column is true label's (i.e. 1)
        """
        if isinstance(X, pd.DataFrame):
            X = np.asarray(X)
        return np.ones((X.shape[0], 2)) * [1 - self.value, self.value]

    def predict_proba(self, X):
        """Reture probability score of each protein on each HPO term.
        :param X: feature matrix (actually no use)
        :return: predictive scores, see predict()
        """
        return self.predict(X)


class FlatModel:
    def __init__(self, model):
        self._model = model
        self._classifiers = dict()

    def _get_model(self):
        """Return model prototype you need.
        :return: model prototype
        """
        if self._model == "lr":
            return LogisticRegression(solver='liblinear')
        else:
            raise ValueError("Can't recognize the model %s" % self._model)

    def _fit_job(self, X, annotation, term_list):
        """Sub-routine of fit classifiers of HPO terms in term_list.
        :param X: dense numpy array, feature matrix
        :param annotation: pandas DataFrame, HPO annotations
        :param term_list: list of HPO terms to be fitted
        :return: dict, hpo term -> classifier
        """
        # transform dense array to sparse matrix when the density < 0.7
        # else leave it as is
        X = array_to_csr(X)
        # dict of classifier of each HPO term in term_list
        classifiers = dict()
        for hpo_term in term_list:
            # extract labels
            y = np.asarray(annotation[[hpo_term]])[:, 0]
            # choose proper model
            if len(np.unique(y)) == 2:
                clf = self._get_model()
            else:
                clf = SameModel()
            # train the classifier
            clf.fit(X, y)
            # store the model
            classifiers[hpo_term] = clf
            # output the logging message
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                  "Fit", hpo_term)
        return classifiers

    def fit(self, feature, annotation, num_worker=1):
        """Fit the model according to the given feature and HPO annotations.

        N.B. The number of proteins in feature and annotation are MUST be the
            SAME!!!
        :param feature: features, DataFrame instance with rows being proteins
            and columns being HPO terms, the values are real number
        :param annotation: HPO annotations, DataFrame instance with rows being
            proteins and columns being HPO terms, the values are 0/1
        :param num_worker: number of sub-processes to fit the classifiers
        :return: None
        """
        assert isinstance(feature, pd.DataFrame), \
            "Argument feature must be Pandas DataFrame instance."
        assert isinstance(annotation, pd.DataFrame), \
            "Argument annotation must be Pandas DataFrame instance."
        assert feature.shape[0] == annotation.shape[0], \
            "The number of proteins in feature and annotation are must be " \
            "the same."

        term_list = annotation.columns
        X = feature.values
        if num_worker > 1:
            # share the memory space of feature matrix
            # X = sharedmem.copy(X)
            X=X
            # start parallel processing by segmenting several sub-sequences
            res_ids = list()
            pool = Pool(num_worker)
            for i in range(num_worker):
                res = pool.apply_async(
                    self._fit_job,
                    args=(X, annotation, term_list[i::num_worker])
                )
                res_ids.append(res)
            pool.close()
            pool.join()
            # combine the results
            self._classifiers = reduce(
                lambda a, b: {**a, **b},
                [res_id.get() for res_id in res_ids]
            )
        else:
            self._classifiers = self._fit_job(X, annotation, term_list)

    def predict(self, feature):
        """Predict scores on each HPO terms according to given features.
        :param feature: features, DataFrame instance with rows being proteins
            and columns being HPO terms, the values are real number
        :return: predictive score, dict like
        { protein1: { term1: score1, term2: score2
        """
        assert isinstance(feature, pd.DataFrame), \
            "Argument feature must be Pandas DataFrame instance."

        score = defaultdict(dict)
        protein_list = feature.axes[0].tolist()
        for hpo_term in self._classifiers:
            clf = self._classifiers[hpo_term]
            prediction = clf.predict_proba(feature)[:, 1]
            for idx, protein in enumerate(protein_list):
                score[protein][hpo_term] = prediction[idx]
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                  "Predict", hpo_term)

        return score


if __name__ == "__main__":


    conf_path = "../../../config/basic/flat/flat_text-fusion_all_full_temporal_22.json"
    if len(sys.argv) > 1 and len(sys.argv[1]) > 0:
        conf_path = sys.argv[1]

    with open(conf_path) as fp:
        config = json.load(fp)

    # load HPO
    ontology = HumanPhenotypeOntology(config["ontology"]["path"],
                                      version=config["ontology"]["version"])
    # get namespace id list
    ns_id = get_ns_id(version=config["ontology"]["version"])

    # load training set, ltr training set and test set
    train_protein_list = load_protein(config["protein_list"]["train"])
    ltr_protein_list = load_protein(config["protein_list"]["ltr"])
    test_protein_list = load_protein(config["protein_list"]["test"])

    # load features and convert them to DataFrame
    # feature = load_feature(config["feature"])
    df_feature = pd.DataFrame.from_dict(load_feature(config["feature"]), orient="index").fillna(0)
    # df_feature = df_feature.fillna(0)
    # protein_list = list(df_feature.index)
    # train_x = torch.from_numpy(df_feature.values)
    # train_x = train_x.float()
    # feature_size = train_x.shape[1]
    #
    # EPOCH = 5
    # LR = 0.001
    # BATCH_SIZE = 100
    #
    # loader = DataLoader(dataset=train_x, batch_size=BATCH_SIZE, shuffle=True)
    #
    # Coder = AutoEncoder()
    #
    # optimizer = torch.optim.Adam(Coder.parameters(), lr=LR)
    # loss_func = nn.MSELoss()
    #
    # for epoch in range(EPOCH):
    #     for step, x in enumerate(loader):
    #         encoded, decoded = Coder(x)
    #         loss = loss_func(decoded, x)
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #
    #         if step % 10 == 0:
    #             print("Epoch: ", epoch, "|", "train_loss: %.4f" % loss.data)
    #
    # train_x, _ = Coder(train_x)
    # train_x = train_x.detach().numpy()
    # # df_feature = pd.DataFrame(train_x, index=protein_list)
    # df_feature = pd.concat([df_feature, pd.DataFrame(train_x, index=protein_list)], axis=1)

    combined_ltr_result = defaultdict(dict)
    combined_test_result = defaultdict(dict)
    # load propagated HPO annotations of specified sub-ontology
    hpo_annotation = load_annotation(config["annotation"], ontology, ns='all')
    # transform it to double-layer dict, i.e.
    # { protein1: { hpo_term1: score1, hpo_term2: score2, ... }, ... }
    dict_annotation = defaultdict(dict)
    for protein in hpo_annotation:
        for hpo_term in hpo_annotation[protein]:
            dict_annotation[protein][hpo_term] = 1
    # convert annotation to DataFrame
    df_annotation = pd.DataFrame.from_dict(dict_annotation, orient="index")
    df_annotation = df_annotation.fillna(0)

    # extract training features and annotations
    train_protein_of_ns = list(set(train_protein_list) &
                               set(df_annotation.axes[0]) &
                               set(df_feature.axes[0]))
    train_feature = df_feature.loc[train_protein_of_ns]
    train_annotation = df_annotation.loc[train_protein_of_ns]
    # extract ltr training features and annotations
    ltr_protein_of_ns = list(set(ltr_protein_list) &
                             set(df_feature.axes[0]))
    ltr_feature = df_feature.loc[ltr_protein_of_ns]
    # extract test features and annotations
    test_protein_of_ns = list(set(test_protein_list) &
                              set(df_feature.axes[0]))
    test_feature = df_feature.loc[test_protein_of_ns]

    # train model and predict
    classifier = FlatModel(model=config["model"])
    classifier.fit(train_feature, train_annotation)
    ltr_result = classifier.predict(ltr_feature)
    test_result = classifier.predict(test_feature)

    with open(config["result"]["ltr"], "w") as fp:
        json.dump(ltr_result, fp)#, orient='index')# indent=2, orient='index')
    with open(config["result"]["test"], "w") as fp:
        json.dump(test_result, fp)#, orient='index')# indent=2, orient='index')

    #
    # for ns in ns_id:
    #     # load propagated HPO annotations of specified sub-ontology
    #     hpo_annotation = load_annotation(config["annotation"], ontology, ns)
    #     if len(hpo_annotation) == 0:
    #         continue
    #     # transform it to double-layer dict, i.e.
    #     # { protein1: { hpo_term1: score1, hpo_term2: score2, ... }, ... }
    #     dict_annotation = defaultdict(dict)
    #     for protein in hpo_annotation:
    #         for hpo_term in hpo_annotation[protein]:
    #             dict_annotation[protein][hpo_term] = 1
    #     # convert annotation to DataFrame
    #     df_annotation = pd.DataFrame.from_dict(dict_annotation, orient="index")
    #     df_annotation = df_annotation.fillna(0)
    #
    #     # extract training features and annotations
    #     train_protein_of_ns = list(set(train_protein_list) &
    #                                set(df_annotation.axes[0]) &
    #                                set(df_feature.axes[0]))
    #     train_feature = df_feature.loc[train_protein_of_ns]
    #     train_annotation = df_annotation.loc[train_protein_of_ns]
    #     # extract ltr training features and annotations
    #     ltr_protein_of_ns = list(set(ltr_protein_list) &
    #                              set(df_feature.axes[0]))
    #     ltr_feature = df_feature.loc[ltr_protein_of_ns]
    #     # extract test features and annotations
    #     test_protein_of_ns = list(set(test_protein_list) &
    #                               set(df_feature.axes[0]))
    #     test_feature = df_feature.loc[test_protein_of_ns]
    #
    #     # train model and predict
    #     classifier = FlatModel(model=config["model"])
    #     classifier.fit(train_feature, train_annotation)
    #     ltr_result = classifier.predict(ltr_feature)
    #     test_result = classifier.predict(test_feature)
    #
    #     # combine result into final result
    #     for protein in ltr_result:
    #         for term in ltr_result[protein]:
    #             combined_ltr_result[protein][term] = ltr_result[protein][term]
    #     for protein in test_result:
    #         for term in test_result[protein]:
    #             combined_test_result[protein][term] = test_result[protein][term]
    #
    # # write result
    # with open(config["result"]["ltr"], "w") as fp:
    #     json.dump(combined_ltr_result, fp, indent=2)
    # with open(config["result"]["test"], "w") as fp:
    #     json.dump(combined_test_result, fp, indent=2)
