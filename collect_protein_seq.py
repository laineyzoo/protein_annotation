
import csv
import numpy as np
import json
from Bio import Entrez
from Bio import ExPASy
from Bio import SwissProt
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


#get all accession numbers of our dataset
f = open("protein_records_all.csv","r")
reader = csv.reader(f)
data = np.array(list(reader))
f.close()
accessions = list(set(data[:,0]))

#get the protein sequences
sequence_dict = {}
sequence_list = []

for i in range(len(accessions)):
	access = accessions[i]
	handle = ExPASy.get_sprot_raw(access)
	record = SwissProt.read(handle)
	seq = record.sequence
	sequence_list.append((access, seq))
	sequence_dict[access] = seq
	print("Sequence list: ", len(sequence_list))

with open("protein_sequences.json","w") as f:
	json.dump(sequence_dict, f)

with open("protein_sequences.csv","w") as f:
	writer = csv.writer(f)
	for row in sequence_list:
		writer.writerow(row)

print("Done writing mRNA sequences to file")

