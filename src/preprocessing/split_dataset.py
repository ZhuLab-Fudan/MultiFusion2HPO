import json
import random
from collections import defaultdict
from functools import reduce
from ontology import HumanPhenotypeOntology
from ontology import get_root, get_subontology

with open("../../config/preprocessing/split_dataset.json") as fp:
    config = json.load(fp)

with open(config["raw_annotation"]) as fp:
    annotation = json.load(fp)
random.seed(0)
protein_list = list(annotation.keys())
random.shuffle(protein_list)

ontology = HumanPhenotypeOntology(config["ontology"]["path"], version=config["ontology"]["version"])
# k = config["k"]
#
# for i in range(k):
#     if i == k - 1:
#         basic_protein_list = protein_list[: int(len(protein_list) * i / k)]
#         ltr_test_protein_list = protein_list[int(len(protein_list) * i / k):]
#     else:
#         ltr_test_protein_list = protein_list[int(len(protein_list) * i / k): int(len(protein_list) * (i + 1) / k)]
#         basic_protein_list = protein_list[0: int(len(protein_list) * i / k)] + protein_list[int(len(protein_list) * (i + 1) / k):]
#     ltr_test_protein_list_p1 = ltr_test_protein_list[0: int(len(ltr_test_protein_list) / 2)]
#     ltr_test_protein_list_p2 = ltr_test_protein_list[int(len(ltr_test_protein_list) / 2):]
#
#     basic_annotation = {p: annotation[p] for p in basic_protein_list}
#     ltr_test_annotation_p1 = {p: annotation[p] for p in ltr_test_protein_list_p1}
#     ltr_test_annotation_p2 = {p: annotation[p] for p in ltr_test_protein_list_p2}
#
#     with open(config["processed_annotation_prefix"]["train"] + str(i) + ".json", 'w') as fp:
#         json.dump(basic_annotation, fp, indent=2)
#     with open(config["processed_annotation_prefix"]["ltr_test"] + str(i) + "_1.json", 'w') as fp:
#         json.dump(ltr_test_annotation_p1, fp, indent=2)
#     with open(config["processed_annotation_prefix"]["ltr_test"] + str(i) + "_2.json", 'w') as fp:
#         json.dump(ltr_test_annotation_p2, fp, indent=2)
#     with open(config["protein_list_prefix"]["train"] + str(i) + ".json", 'w') as fp:
#         json.dump(basic_protein_list, fp)
#     with open(config["protein_list_prefix"]["ltr_test"] + str(i) + "_1.json", 'w') as fp:
#         json.dump(ltr_test_protein_list_p1, fp)
#     with open(config["protein_list_prefix"]["ltr_test"] + str(i) + "_2.json", 'w') as fp:
#         json.dump(ltr_test_protein_list_p2, fp)

propagated_annotation = dict()
for protein in annotation:
    propagated_annotation[protein] = list(ontology.transfer(annotation[protein]) - {get_root()} - set(get_subontology(ontology.version)))

with open(config["term_list"], 'w') as fp:
    json.dump(list(reduce(lambda a, b: set(a) | set(b), propagated_annotation.values())), fp, indent=2)

term_counts = defaultdict(int)
for protein in propagated_annotation:
    for term in propagated_annotation[protein]:
        term_counts[term] += 1

for interval in config["frequency"]:
    with open(interval["path"], 'w') as fp:
        json.dump([term for term in term_counts if interval["low"] <= term_counts[term] < interval["high"]], fp, indent=2)
