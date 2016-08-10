
import csv
import numpy as np
import json
from Bio import Entrez
from sets import Set

Entrez.email = "elaine.zosa@helsinki.fi"

GO_JSON = "go.json"
f = open(GO_JSON)
go_ontology = json.load(f)
f.close()

GO_PARENTS = "go_parents.json"
f = open(GO_PARENTS)
go_parents = json.load(f)
f.close()

GO_CHILDREN = "go_children.json"
f = open(GO_CHILDREN)
go_children = json.load(f)
f.close()

def get_children(go_term):
	return go_children[go_term]

def get_parents(go_term):
	return go_parents[go_term]


#get the PubMed IDs of the papers we want
pubmed_ids = list()
go_terms = list()
for node in go_ontology:
	node_id = node['id']
	children = get_children(node_id)
	if len(children) == 0:
		protein_name = node['name']
		handle = Entrez.esearch(db="pubmed", term=protein_name)
		record = Entrez.read(handle)
		pmids = record["IdList"]
		for pmid in pmids:
			pubmed_ids.append(pmid)
			go_terms.append(node_id)
		print("PubMed IDs so far: ", len(pubmed_ids))

total = len(pubmed_ids)
print("Total PubMed IDs: ", total)
#get the paper abstracts
web_history = Entrez.read(Entrez.epost("pubmed", id=",".join(pubmed_ids)))
web_env = web_history["WebEnv"]
query_key = web_history["QueryKey"]

pubmed_records = list()
for i in range(219505,len(pubmed_ids)):
	pmid = pubmed_ids[i]
	go_term = go_terms[i]
	print("Fetching " + str(i) + " of " + str(total))
	handle = Entrez.efetch(db="pubmed", id=pmid, rettype="abstract", retmode="text", webenv=web_env, querykey=query_key)
	abstract = handle.read()
	abstract = abstract.replace("\n"," ")
	tup = (pmid, abstract)
	pubmed_records.append(tup)
	count = len(pubmed_records)
	print("Records = ", count)

fname = "pubmed_papers_leaves.csv"
with open(fname, "w") as f:
	writer = csv.writer(f)
	for row in pubmed_records:
		writer.writerow(row)

print("Done writing all papers.")

#travers ontology bottom-up tp assign papers to non-leaf nodes
papers_dict = {}
for i in range(len(go_terms)):
	go_id = go_terms[i]
	pmid = pubmed_ids[i]
	if go_id in papers_dict.keys():
		papers_dict[go_id].append(pmid)
	else:
		papers_dict[go_id] = []
		papers_dict[go_id].append(pmid)
leaf_nodes = papers_dict.keys()
q = collections.deque()
q.extend(leaf_nodes)

while len(q)>0:
	node = q.popleft()
	parents = get_parents(node)
	for p in parents:
		if p not in papers_dict.keys():
			papers_dict[p] = []
		papers_list = list(set(papers_dict[p] + papers_dict[node]))
		papers_dict[p].extend(papers_list)
		q.append(p)

