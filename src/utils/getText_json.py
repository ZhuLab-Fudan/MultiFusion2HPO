import time
import random
import requests
import re
import json
import threading
import pickle
from pubmed_lookup import PubMedLookup, Publication

pt_lock = threading.Lock()
pl_lock = threading.Lock()

with open("uniprot-compressed_true_download_true_format_json_query_all_protein_2023.json") as fp:
    file = json.load(fp)
protein_list = []
protein_pmids = {}
for result in file["results"]:
    protein = result["primaryAccession"]
    pmids = []
    for reference in result["references"]:
        pmids.append(reference["citation"]["id"])
    protein_list.append(protein)
    protein_pmids[protein] = pmids

email = "2011085010@fudan.edu.cn"

protein_text = {}
protein_file = {}
count = 0

# protein_list = ["Q9BQE9"]

class GetText_Parse_XML(threading.Thread):
    def run(self):
        global pt_lock
        global pl_lock
        # global email
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
            PMIDs = protein_pmids[protein]
            if PMIDs == []:
                pt_lock.acquire()
                protein_text.update({protein: []})
                pt_lock.release()
                continue

            print(protein + ": " + str(len(PMIDs)))
            text = []
            # for PMID in PMIDs:
            #     try:
            #         url = 'http://www.ncbi.nlm.nih.gov/pubmed/' + PMID
            #         lookup = PubMedLookup(url, email)
            #         publication = Publication(lookup)
            #         text.append(publication)
            #         print("error process pmid " + PMID + " for protein " + protein)
            #     except Exception:
            #         continue
            # protein_text.update({protein: text})
            for PMID in PMIDs:
                try:
                    time.sleep(random.randint(0, 10) / 10.0)
                    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&api_key=61af0e85c0f1ba3577fba898d7e8c73de509&id=" + PMID
                    publication = requests.get(url)
                    if publication.status_code == 200:
                        text.append(publication.text)
                        print(protein + ": " + PMID)
                    else:
                        print(publication.text)
                except:
                    print("error process pmid " + PMID + " for protein " + protein)
                    continue
            pt_lock.acquire()
            protein_text.update({protein: text})
            pt_lock.release()

# 格式化，但是速度较慢
class GetText_Pubmed(threading.Thread):
    def run(self):
        global pt_lock
        global pl_lock
        # global email
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
            PMIDs = protein_pmids[protein]
            if PMIDs == []:
                pt_lock.acquire()
                protein_text.update({protein: []})
                pt_lock.release()
                continue

            print(protein + ": " + str(len(PMIDs)))
            text = []
            for PMID in PMIDs:
                try:
                    url = 'http://www.ncbi.nlm.nih.gov/pubmed/' + PMID
                    lookup = PubMedLookup(url, email)
                    publication = Publication(lookup)
                    text.append(publication)
                    print("error process pmid " + PMID + " for protein " + protein)
                except Exception:
                    continue
            protein_text.update({protein: text})
            for PMID in PMIDs:
                time.sleep(random.randint(0, 10) / 10.0)
                url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&api_key=61af0e85c0f1ba3577fba898d7e8c73de509&id=" + PMID
                publication = requests.get(url)
                if publication.status_code == 200:
                    text.append(publication.text)
                    print("error process pmid " + PMID + " for protein " + protein)
                else:
                    print(publication.text)
            #         print(protein + " " + PMID)
            pt_lock.acquire()
            protein_text.update({protein: text})
            pt_lock.release()


def decode(text):
    text = re.sub("\s+", ' ', text)
    pmid = int(re.findall(">(\d+)</PMID>", text)[0])
    year = int(re.findall("<Year>(\d{4})</Year>", text)[0])
    month = int(re.findall("<Month>(\d{1,2})</Month>", text)[0])
    day = int(re.findall("<Day>(\d{1,2})</Day>", text)[0])
    abstract = re.findall("<AbstractText>(.*?)</AbstractText>", text)[0]
    return {pmid: {"year": year, "month": month, "day": day, "abstract": abstract}}

threads = []
for i in range(10):
    threads.append(GetText_Parse_XML())
for thread in threads:
    thread.start()
for thread in threads:
    thread.join()

for protein in protein_text:
    file = {}
    for text in protein_text[protein]:
        try:
            file.update(decode(text))
        except:
            print("except text: " + text)
    protein_file.update({protein: file})

# with open("all_protein_text.pkl", 'wb') as fp:
#     pickle.dump(protein_text, fp)
#
# with open("all_protein_file.json", 'w') as fp:
#     json.dump(protein_file, fp, indent=3)
with open("all_protein_text_2023.pkl", 'wb') as fp:
    pickle.dump(protein_text, fp)

with open("all_protein_file_2023.json", 'w') as fp:
    json.dump(protein_file, fp, indent=3)