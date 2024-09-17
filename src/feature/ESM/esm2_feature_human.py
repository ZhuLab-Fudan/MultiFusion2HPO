import json
from Bio import SwissProt
from transformers import EsmModel, EsmTokenizer
import torch

# 加载ESM2模型和tokenizer
# model = EsmModel.from_pretrained("facebook/esm2_t6_8M_UR50D")
# tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
model = EsmModel.from_pretrained("/public/home/zhaiwq/modelused/DeepText2HPO_pycharm_project/data/feature/esm2/esm2_t36_3B_UR50D")
tokenizer = EsmTokenizer.from_pretrained("/public/home/zhaiwq/modelused/DeepText2HPO_pycharm_project/data/feature/esm2/esm2_t36_3B_UR50D")

def generate_protein_representation(sequence):
    inputs = tokenizer(sequence, return_tensors="pt", add_special_tokens=False)
    with torch.no_grad():
        outputs = model(**inputs)
    # 提取平均特征表示
    protein_representation = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
    return protein_representation

def read_dat(file_path):
    protein_dict = {}
    with open(file_path) as handle:
        for record in SwissProt.parse(handle):
            if "Homo sapiens" in record.organism:  # 检查物种字段
                entry_name = record.entry_name
                sequence = record.sequence
                protein_representation = generate_protein_representation(sequence)
                protein_dict[entry_name] = protein_representation
    return protein_dict

def save_to_json(data, file_path):
    with open(file_path, 'w') as json_file:
        json.dump(data, json_file)

# 文件路径
dat_file_path = "/public/home/zhaiwq/modelused/DeepText2HPO_pycharm_project/data/uniprot_sprotkb/uniprot_sprot-only2021_04/uniprot_sprot.dat"
output_json_path = "human_protein_representations.json"

# 读取DAT文件并生成特征表示
protein_dict = read_dat(dat_file_path)

# 保存结果为JSON文件
save_to_json(protein_dict, output_json_path)

print(f"人类蛋白质的特征表示已保存至 {output_json_path}")


