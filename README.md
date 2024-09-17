# MultiFusion2HPO

## 如何运行？

注：请在开始下面步骤前，在~/.bashrc文件的最后一行加上：

export PYTHONPATH=${PYTHONPATH}:[MultiFusion2HPO的目录位置]


### 第一步：预处理(Pre-processing)

1. 从http://compbio.charite.de/jenkins/job/hpo.annotations.monthly/下载基因-HPO标注文件

ALL_SOURCES_ALL_FREQUENCIES_genes_to_phenotype.txt到目录data/annotation/raw。然后，运行src/preprocessing/extract_gene_id.py以提取文件的第一列（即entrez gene id），并将其写入一个.txt文件。接着，上传该文件内容到Uniprot ID Mapping Tool (http://www.uniprot.org/mapping/)，选择选项From: Entrez Gene (Gene ID)和To: UniProtKB，然后点击Submit按钮。在新的窗口中，选择左栏的Filter by为Reviewed Swiss-Prot，并点击页面中间的Download按钮，选择Format: Tab-separated并点击Go下载映射文件。将映射文件放入data/annotation/intermediate中，并更名为gene2uniprot.txt。

2. 运行src/preprocessing/create_annotation.py程序。程序将输出处理好的HPO标注文件。

3. 重复上述两步，分别处理得到三个标注文件(比如2021-04-13、2022-04-14和2024-04-26)。

4. 从https://bioportal.bioontology.org/ontologies/HP下载三个时期相对应的.obo文件，将其放在data/obo目录下。之后运行src/preprocessing/split_dataset.py。我们将得到：

    处理好的用于训练基础分类器的标注、用于训练排序学习的标注和用于测试的标注

    上面三个标注文件中的蛋白质标识符列表

    用来标注蛋白质的HPO term列表

    按照频率划分的各个小组内的HPO term列表

### 第二步：处理特征（Extracting Features）

#### TF-IDF-D2V

   1. src/utils下的getText.py，使用蛋白质列表获取protein - text abstract对应，输出的.pkl是原始语料，.json经过简单日期整理。

   2. src/utils下的textCombine.py，用于获取protein与经过去停用词、词形还原等操作之后的text-abstract的关联，每个protein关联的多个text-abstract进行拼接。

   3. 获取到的protein - text文件，运行src/feature/TF-IDF/tf-idf.py和src/feature/D2V/d2v.py程序，得到在data/feature/TF-IDF/clean目录和data/feature/D2V/clean目录下的json格式的文件。

   4. 运行src/feature/Text-Fusion/text-fusion.py程序，得到在data/feature/Text_fusion/clean目录下的json格式的文件。

#### BioLinkBERT

   1.去往https://huggingface.co/michiyasunaga/BioLinkBERT-large下载BioLinkBERT-large模型。

   2. 利用上述得到的的protein - text文件运行src/feature/BioLinkBERT/biolinkbert_feature.py程序，得到在data/feature/BioLinkBERT/clean目录下的json格式的文件。


#### STRING

   1. 打开https://string-db.org/，然后点击页面左上角的Version，在自动跳转到的新页面中选择合适的版本，并单击Address一栏中的链接。之后，点击新页面上方导栏的Download按钮，点击choose an organism下拉菜单，选择Homo sapiens。现在，点击INTERACTION DATA部分的9606.protein.links.XXX.txt.gz（XXX为版本），下载蛋白质互作数据。最后，点击ACCESSORY DATA部分的mapping_files (download directory)，进入ftp页面，点击uniprot_mappings/目录，下载属于人类（可能是开头为9606或者文件名中有human字样）的压缩文件。上述两个文件都下载至data/feature/STRING/raw目录下。

   2. 运行src/feature/STRING/string.py程序，得到在data/feature/STRING/clean目录下的json格式的文件。

#### GO Annotation

   1. 首先，在https://www.ebi.ac.uk/GOA/downloads页面的Annotation Sets表格的Human一行中，点击某一个链接（若要下载当前最新数据，点击Current Files，否则点击Archive Files），然后下载合适版本的.gaf文件。

   2. 若要下载最新版的GO的.obo文件，可从http://geneontology.org/docs/download-ontology/中下载；若要下旧版，则可以从https://bioportal.bioontology.org/ontologies/GO下载。

   3. 运行src/feature/GO_annotation/go_annotation.py，得到在data/feature/GO_annotation/clean目录下的数据。


#### ESM2

   1. 打开https://huggingface.co/facebook/esm2_t36_3B_UR50D，文件都下载至data/feature/ESM/raw目录下。

   2. 运行src/feature/ESM/esm2_feature_human.py程序，得到在data/feature/ESM/clean目录下的json格式的文件。


#### Gene-Expression

   1. 打开https://github.com/bio-ontology-research-group/deeppheno/，Expression Atlas Genotype-Tissue Expression (GTEx) Project (E-MTAB-5214) 文件都下载至data/feature/Gene_expression/raw目录下。

   2. 运行src/feature/STRING/gene_expression.py程序，得到在data/feature/Gene_expression/clean目录下的json格式的文件。


#### InterPro

   1. 从http://ftp.ebi.ac.uk/pub/software/unix/iprscan/5/选择合适的InterProScan程序包下载。

   2. 进入下载后解压的目录内，以要查询的蛋白质的fasta文件为输入，运行InterProScan，得到程序匹配到的InterPro signatures的xml文件：

      ./interproscan.sh -i /path/to/sequences.fasta -b /path/to/output_file -f XML

   3. 运行src/feature/InterPro/interpro.py程序，处理上一步得到的原始xml文件，获得处理后的InterPro特征文件。



### 第三步：训练基础分类器（Basic Models）

#### Naive

   1. 运行src/basic/naive/naive.py，得到data/result/basic/naive内的输出文件。

#### Neighbor

   1. 注意设置配置文件里“network”一项的“type”，对于STRING，其设为“weighted。

   2. 运行src/basic/neighbor/neighbor.py，得到保存在data/result/basic/neighbor中的预测结果。

#### Flat

   1. 运行src/basic/flat/flat.py，将各种处理得到的特征文件作为输入，训练Logistic Regression分类器，对用于排序学习的训练集和测试集进行预测，得到输出在data/result/basic/flat目录下的一系列预测结果文件。

#### DNN

   1. 运行src/basic/dnn/dnn.py，将各种处理得到的特征文件作为输入，训练DNN分类器，对用于排序学习的训练集和测试集进行预测，得到输出在data/result/basic/dnn目录下的一系列预测结果文件。


### 第四步：排序学习（Learning to Rank）

   1. 将第三步中得到的一系列预测分数作为输入，即配置文件中的"result"部分。注意：请务必保证这一部分的"ltr"和"test"的列表内的文件顺序是一致的！

   2. 注意合理调节配置文件中"model_param"部分的max_depth值、"fit_param"部分中的num_boost_round和early_stopping_rounds以及"top"中的取值。

   3. 运行src/ensemble/ltr/ltr.py程序，得到最终的预测结果。

### 第五步：评估（Evaluation）

   1. 将要评估的预测结果的文件路径添加在配置文件的"result"部分。

   2. 运行src/utils/evaluation.py，程序将会显示各个预测结果在各个子本体上的

       Fmax：以蛋白质为中心的评估指标

       AUC：以HPO term为中心的评估指标，即每个HPO term的AUC的平均值

       AUPR：整体的评估指标，即以一对蛋白质-HPO term为实例进行计算的AUPR

       每个按频率划分的HPO term小组内的平均AUC
