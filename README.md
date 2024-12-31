# MultiFusion2HPO

## How To Run？

Note: Please add the last line of the ~/.bashrc file before starting the following steps:

export PYTHONPATH=${PYTHONPATH}:[Directory location of MultiFusion2HPO]


### Step 1: Pre-processing

1. Download the Gene-HPO Annotation file from http://compbio.charite.de/jenkins/job/hpo.annotations.monthly/

ALL_SOURCES_ALL_FREQUENCIES_genes_to_phenotype.txt to the directory data/annotation/raw. Then, run src/preprocessing/extract_gene_id.py to extract the first column of the file (the entrez gene id) and write it to a .txt file. Next, upload the contents of this file to the Uniprot ID Mapping Tool (http://www.uniprot.org/mapping/), select the options From: Entrez Gene (Gene ID) and To: UniProtKB, and click the Submit button. In the new window, select Filter by for Reviewed Swiss-Prot in the left column and click the Download button in the middle of the page, select Format: Tab-separated and click Go to download the mapping file. Put the mapping file in data/annotation/intermediate and rename it to gene2uniprot.txt.

2. Run the src/preprocessing/create_annotation.py. It will output the processed HPO annotation files.

3. Repeat the two steps above to process each to get three annotation files (e.g., 2021-04-13, 2022-04-14, and 2024-04-26).

4. Download the .obo files corresponding to the three periods from https://bioportal.bioontology.org/ontologies/HP and put them in the data/obo directory. Then run src/preprocessing/split_dataset.py. We will get:

    Processed annotations for training the base classifier, annotations for training sort learning and annotations for testing

    List of protein identifiers in the three annotation files above

    List of HPO terms used to annotate proteins

    List of HPO terms within each group by frequency

### Step2：Extracting Features

#### TF-IDF-D2V

   1. Run getText.py in src/utils to obtain the protein-text abstract correspondence using the protein list. The output .pkl is the original corpus, and .json is simply sorted by date.

   2. Run textCombine.py in src/utils to obtain the association between protein and text-abstract after operations such as removing stop words and restoring morphology. Multiple text-abstracts associated with each protein are spliced.

   3. After obtaining the protein-text file, run src/feature/TF-IDF/tf-idf.py and src/feature/D2V/d2v.py programs to obtain json format files in the data/feature/TF-IDF/clean directory and data/feature/D2V/clean directory.

   4. Run src/feature/Text-Fusion/text-fusion.py program to obtain json format files in the data/feature/Text_fusion/clean directory.

#### BioLinkBERT

   1. Go to https://huggingface.co/michiyasunaga/BioLinkBERT-large to download the BioLinkBERT-large model.

   2. Use the protein-text file obtained above to run the src/feature/BioLinkBERT/biolinkbert_feature.py program to obtain a json file in the data/feature/BioLinkBERT/clean directory.

#### STRING

   1. Open https://string-db.org/, then click Version in the upper left corner of the page, select the appropriate version in the new page that automatically jumps to, and click the link in the Address column. After that, click the Download button in the top bar of the new page, click the choose an organism drop-down menu, and select Homo sapiens. Now, click 9606.protein.links.XXX.txt.gz (XXX is the version) in the INTERACTION DATA section to download the protein interaction data. Finally, click mapping_files (download directory) in the ACCESSORY DATA section, enter the ftp page, click the uniprot_mappings/ directory, and download the compressed file belonging to humans (it may start with 9606 or have the word human in the file name). Both of the above files are downloaded to the data/feature/STRING/raw directory.

   2. Run the src/feature/STRING/string.py program to get the json format files in the data/feature/STRING/clean directory.

#### GO Annotation

   1. First, in the Human row of the Annotation Sets table on the https://www.ebi.ac.uk/GOA/downloads page, click a link (to download the latest data, click Current Files, otherwise click Archive Files), and then download the appropriate version of the .gaf file.

   2. To download the latest version of GO's .obo file, you can download it from http://geneontology.org/docs/download-ontology/; if you want to download an older version, you can download it from https://bioportal.bioontology.org/ontologies/GO.

   3. Run src/feature/GO_annotation/go_annotation.py to get the data in the data/feature/GO_annotation/clean directory.


#### ESM2

   1. Open https://huggingface.co/facebook/esm2_t36_3B_UR50D and download all the files to the data/feature/ESM/raw directory.

   2. Run the src/feature/ESM/esm2_feature_human.py program to get the json format file in the data/feature/ESM/clean directory.


#### Gene-Expression

   1. Open https://github.com/bio-ontology-research-group/deeppheno/ and download all Expression Atlas Genotype-Tissue Expression (GTEx) Project (E-MTAB-5214) files to the data/feature/Gene_expression/raw directory.

   2. Run the src/feature/STRING/gene_expression.py program to get the json format files in the data/feature/Gene_expression/clean directory.


#### InterPro

   1. Download the appropriate InterProScan package from http://ftp.ebi.ac.uk/pub/software/unix/iprscan/5/.

   2. Enter the directory where you unzipped the downloaded file, use the fasta file of the protein to be queried as input, run InterProScan, and obtain the XML file of the InterPro signatures matched by the program:

      ./interproscan.sh -i /path/to/sequences.fasta -b /path/to/output_file -f XML

   3. Run the src/feature/InterPro/interpro.py program to process the original XML file obtained in the previous step and obtain the processed InterPro feature file.


### Step3：Train Basic Models

#### Naive

   1. Run src/basic/naive/naive.py and get the output file in data/result/basic/naive.

#### Neighbor

   1. Pay attention to the "type" of the "network" item in the configuration file. For STRING, it is set to "weighted.

   2. Run src/basic/neighbor/neighbor.py and get the prediction results saved in data/result/basic/neighbor.

#### Flat

   1. Run src/basic/flat/flat.py, use the feature files obtained from various processes as input, train the Logistic Regression classifier, predict the training set and test set used for ranking learning, and output a series of prediction result files in the data/result/basic/flat directory.

#### DNN

   1. Run src/basic/dnn/dnn.py, use the feature files obtained by various processing as input, train the DNN classifier, predict the training set and test set used for sorting learning, and output a series of prediction result files in the data/result/basic/dnn directory.

### Step4：Learning to Rank

   1. Take the series of prediction scores obtained in the third step as input, that is, the "result" part of the configuration file. Note: Please make sure that the order of the files in the "ltr" and "test" lists in this part is consistent!

   2. Pay attention to reasonably adjust the max_depth value in the "model_param" part of the configuration file, num_boost_round and early_stopping_rounds in the "fit_param" part, and the value in "top".

   3. Run the src/ensemble/ltr/ltr.py program to get the final prediction result.

### Step5：Evaluation


   1. Add the file path of the prediction result to be evaluated to the "result" section of the configuration file.

   2. Run src/utils/evaluation.py, and the program will display the prediction results on each sub-ontology

      Fmax: protein-centric evaluation metric

      AUC: HPO term-centric evaluation metric, that is, the average AUC of each HPO term

      AUPR: overall evaluation metric, that is, the AUPR calculated for a pair of protein-HPO terms

      Average AUC within each frequency-divided HPO term group
