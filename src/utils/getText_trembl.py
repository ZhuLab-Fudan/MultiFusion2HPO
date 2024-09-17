import time
import random
import requests
# from pubmed_lookup import PubMedLookup, Publication
import re
import json
import threading
import pickle

pt_lock = threading.Lock()
pl_lock = threading.Lock()

with open("../../data/feature/Trigram/clean/Trigram.json") as fp:
    trigram = json.load(fp)
protein_list = []
for protein in trigram:
    protein_list.append(protein)

text = {}
pubmed_list = {}
count = 0
batch = 0
accession = None
with open("/home/dengyj/uniprot/uniprot_trembl.dat") as fp:
    for line in fp:
        line = str(line)
        if line.startswith("AC"):
            if count >= 100000:
                for protein in protein_list:
                    if protein in text:
                        pubmed_list[protein] = text[protein]
                text = {}
                count = 0
                batch += 1
                print(batch)
            accession = re.findall("\s+([A-Z0-9]+)?;", line)[0]
            text[accession] = []
            count += 1
            print(accession)
            continue
        if line.startswith("RX"):
            try:
                text[accession].append(re.findall("PubMed=(\d+)?;", line)[0])
            except:
                print(line)
                continue
for protein in protein_list:
    if protein in text:
        pubmed_list[protein] = text[protein]
# pubmed_list = {protein: text[protein] for protein in protein_list}
email = "happydyj603@126.com"

protein_text = {}
protein_file = {}
count = 0
# for protein in protein_list:
#     url = "https://www.uniprot.org/uniprot/" + protein + "/publications"
#     html = requests.get(url).text
#     PMIDs = re.findall("https://pubmed.ncbi.nlm.nih.gov/(\d+)", html)
#     if PMIDs == []:
#         continue
#
#     print(protein + ": " + str(len(PMIDs)))
#     text = []
#     for PMID in PMIDs:
#         url = 'http://www.ncbi.nlm.nih.gov/pubmed/' + PMID
#         lookup = PubMedLookup(url, email)
#         publication = Publication(lookup)
#         text.append(publication)
#     protein_text.update({protein: text})

class GetText(threading.Thread):
    def run(self):
        global pt_lock
        global pl_lock
        global email
        global protein_text
        global count
        while True:
            pl_lock.acquire()
            if count == len(protein_list):
                pl_lock.release()
                return
            protein = protein_list[count]
            count += 1
            print(count)
            pl_lock.release()
            print(protein + ": " + str(len(pubmed_list[protein])))
            if pubmed_list[protein] == []:
                pt_lock.acquire()
                protein_text.update({protein: []})
                pt_lock.release()
                continue

            text = []
            for PMID in pubmed_list[protein]:
                time.sleep(random.randint(0, 10) / 10.0)
                url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&api_key=61af0e85c0f1ba3577fba898d7e8c73de509&id=" + PMID
                publication = requests.get(url)
                if publication.status_code == 200:
                    text.append(publication.text)
                else:
                    print(publication.text)
                    print(protein + " " + PMID)
            pt_lock.acquire()
            protein_text.update({protein: text})
            pt_lock.release()

def decode(text):
    text = re.sub("\s+", ' ', text)
    pmid = int(re.findall("pmid (\d+)", text)[0])
    year = int(re.findall("em std { year (\d{4})", text)[0])
    month = int(re.findall("em std { year \d{4}, month (\d{1,2})", text)[0])
    day = int(re.findall("em std { year \d{4}, month \d{1,2}, day (\d{1,2})", text)[0])
    try:
        abstract = re.findall("abstract \"(.*?)\",", text)[0]
    except:
        abstract = []
        print(str(pmid) + " abstract empty")
    return {pmid: {"year": year, "month": month, "day": day, "abstract": abstract}}

threads = []
for i in range(10):
    threads.append(GetText())
for thread in threads:
    thread.start()
for thread in threads:
    thread.join()

for protein in protein_text:
    file = {}
    for text in protein_text[protein]:
        file.update(decode(text))
    protein_file.update({protein: file})

with open("protein_text_trembl_201902.pkl", 'wb') as fp:
    pickle.dump(protein_text, fp)

with open("protein_file_trembl_201902.json", 'w') as fp:
    json.dump(protein_file, fp, indent=3)
