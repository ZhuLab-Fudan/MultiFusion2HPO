from sklearn import preprocessing
from sklearn.decomposition import PCA
import pandas as pd
import json

with open("../../../config/feature/Text_fusion/Text_fusion.json") as fp:
    config = json.load(fp)

file1_dict = {}
with open(config["file1"]) as fp:
    file1 = json.load(fp)
    file1 = pd.DataFrame.from_dict(file1, orient='index')
    file1_array = file1.values
    # pca = PCA(n_components=100)
    # file1_proj = pca.fit_transform(file1_array)
    # file1 = pd.DataFrame(file1_proj, index=file1.index)
    file1 = pd.DataFrame(preprocessing.normalize(file1.values, norm='l2'), index=file1.index)
    for index, row in file1.iterrows():
        file1_dict[index] = row.to_list()

file2_dict = {}
with open(config["file2"]) as fp:
    file2 = json.load(fp)
    file2 = pd.DataFrame.from_dict(file2, orient='index')
    file2 = pd.DataFrame(preprocessing.normalize(file2.values, norm='l2'), index=file2.index)
    for index, row in file2.iterrows():
        file2_dict[index] = row.to_list()

file = {}
for protein in file1_dict:
    if protein in file2_dict:
        file1_dict[protein].extend(file2_dict[protein])
        file[protein] = file1_dict[protein]

with open(config["output"], 'w') as fp:
    json.dump(file, fp)
