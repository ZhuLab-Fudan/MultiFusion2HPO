#!/usr/bin/env python

import numpy as np
import pandas as pd
import json


# gene2protein='gene2prot_20210413.tsv'
gene2protein='idmapping_2024_07_06.tsv'

# From	Entry	Entry Name	Protein names	Gene Names
# ENSG00000000003	O43657	TSN6_HUMAN	Tetraspanin-6 (Tspan-6) (A15 homolog) (Putative NF-kappa-B-activating protein 321) (T245 protein) (Tetraspanin TM4-D) (Transmembrane 4 superfamily member 6)	TSPAN6 TM4SF6 UNQ767/PRO1560



# datadf=pd.read_csv(gene2protein)
gene2prot = {}
# From	Entry	Entry Name	Reviewed
# for i in range(0,datadf.shape[0]):
#     gene=datadf["Entry Name"][i].split("_HUMAN")[0]
#     protein=datadf["Entry"][i]
#     gene2prot[gene] = protein


with open(gene2protein) as f:
    for line in f:
        if line.startswith("From"):
            continue
        it = line.strip().split('\t')

        gene = it[0]
        protein = it[1]
        gene2prot[gene] = protein

expressions_file='E-MTAB-5214-query-results.tpms.tsv'
# Gene ID	Gene Name	Brodmann (1909) area 24	Brodmann (1909) area 9	C1 segment of cervical spinal cord	EBV-transformed lymphocyte	adrenal gland	amygdala	aorta	atrium auricular region	blood	breast	caudate nucleus	cerebellar hemisphere	cerebellum	cerebral cortex	coronary artery	cortex of kidney	ectocervix	endocervix	esophagogastric junction	esophagus mucosa	esophagus muscularis mucosa	fallopian tube	greater omentum	heart left ventricle	hippocampus proper	hypothalamus	liver	lower leg skin	lung	minor salivary gland	nucleus accumbens	ovary	pancreas	pituitary gland	prostate gland	putamen	sigmoid colon	skeletal muscle tissue	small intestine Peyer's patch	spleen	stomach	subcutaneous adipose tissue	substantia nigra	suprapubic skin	testis	thyroid gland	tibial artery	tibial nerve	transformed skin fibroblast	transverse colon	urinary bladder	uterus	vagina
# ENSG00000000003	TSPAN6	6.0	5.0	7.0	0.3	17.0	7.0	11.0	5.0	0.2	36.0	7.0	2.0	3.0	5.0	11.0	15.0	39.0	47.0	10.0	37.0	8.0	30.0	28.0	2.0	6.0	9.0	24.0	11.0	14.0	32.0	7.0	81.0	9.0	51.0	22.0	4.0	11.0	2.0	18.0	10.0	13.0	33.0	6.0	10.0	70.0	22.0	9.0	35.0	20.0	38.0	16.0	33.0	31.0

gene_exp = {}
max_val = 0
N=0
with open(expressions_file) as f:
    for line in f:
        if line.startswith('#') or line.startswith('Gene'):
            continue
        it = line.strip().split('\t')
        gene_name = it[0].upper()
        if gene_name in gene2prot:
            N=N+1
            exp = np.zeros((53,), dtype=np.float32)
            for i in range(len(it[2:])):
                exp[i] = float(it[2 + i]) if it[2 + i] != '' else 0.0
            gene_exp[gene2prot[gene_name]] = (exp / np.max(exp)).tolist()
            # gene_exp[name2gene[gene_name]] = 0

# print(gene_exp)
print(N)
with open("gene_exp.json", "w", ) as f:
    json.dump(gene_exp, f)  # 写为多行


