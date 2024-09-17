from sklearn.feature_extraction.text import TfidfVectorizer
import json

with open("../../../config/feature/TF-IDF/tf-idf.json") as fp:
    config = json.load(fp)

with open(config["abstract"]) as fp:
    abstract = json.load(fp)

protein_list = [i for i in abstract.keys()]
abstracts = [i for i in abstract.values()]

tfidf_vec = TfidfVectorizer()
tfidf_matrix = tfidf_vec.fit_transform(abstracts)
tfidf_array = tfidf_matrix.toarray()

# if the result of tf-idf too much, write the file by self
with open(config["output"], 'w') as fp:
    fp.write("{\n")
    for i in range(len(protein_list)):
        fp.write("\"" + str(protein_list[i]) + "\": ")
        fp.write(str(tfidf_array[i, :].tolist()))
        if i != len(protein_list) - 1:
            fp.write(",\n")
    fp.write("\n}")

# tfidf_feature = dict(zip(protein_list, tfidf_matrix.toarray().tolist()))

# with open(config["output"], 'w') as fp:
#     json.dump(tfidf_feature, fp)