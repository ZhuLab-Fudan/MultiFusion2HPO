import os
import json
import pickle
#
#
# import numpy as np
# import torch
# from tqdm.auto import tqdm
# from transformers import AutoTokenizer, AutoModel



path_result="/public/home/zhaiwq/modelused/DeepText2HPO_pycharm_project/data/feature/esm2"


path="/public/home/zhaiwq/modelused/DeepText2HPO_pycharm_project/data/feature/esm2/protein_data_2023"


pathfile=path
feature = {}

filelist = os.listdir(pathfile)
for file_name in filelist:
    protein_name = str(file_name).split(".")[0]
    with open(pathfile + "/" + file_name,encoding='gbk') as fp:
        dataemb = json.load(fp)
    embedding = dataemb[protein_name]

    feature.update({protein_name: embedding})

with open(path_result + "/esm2_feature_2023.json", 'w') as fp:
    json.dump(feature, fp, indent=2)


