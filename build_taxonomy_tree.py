from __future__ import division
from __future__ import print_function

import numpy as np
import json
import string

##### SILVA dataset #####

#return all descendants of the named rank from the taxonomy
def get_descendants(rank):
	descendants = []
	q = [rank]
	keys = taxonomy.keys()
	while q:
		current_term = q.pop(0)
		descendants.append(current_term)
		for key in keys:
			if current_term in taxonomy[key]["parent"]:
				if key not in descendants:
					q.append(key)
	return list(set(descendants))


f = open("SILVA_123.1_SSURef_tax_silva_trunc.fasta", "r")
silva = list(f)
organisms = list()
for row in silva:
	if row[0]==">":
		if "GGGGGAUAACAGU" not in row:
			organisms.append(row.strip())

silva_dict = {}
for i in range(len(organisms)):
	print("Row ", i)
	row = organisms[i]
	start = row.index(" ")
	accession = row[1:start]
	ranks = row[start+1:]
	ranks = ranks.strip().split(";")
	for j in range(1,len(ranks)):
		ranks[j] = ranks[j-1]+" "+ranks[j]
	silva_dict[accession] = {}
	silva_dict[accession]["class"] = ranks


taxonomy = {}
keys = silva_dict.keys()
for j in range(len(keys)):
	print("Organism ", j)
	key = keys[j]
	ranks = silva_dict[key]
	for i in range(len(ranks)):
		if ranks[i] in taxonomy.keys():
			#rank already exists, update count
			taxonomy[ranks[i]]["count"] += 1
		else:
			#rank is new, create new entry for count/children/parent
			taxonomy[ranks[i]] = {}
			taxonomy[ranks[i]]["count"] = 1
			taxonomy[ranks[i]]["parent"] = []
			taxonomy[ranks[i]]["children"] = []
		if i > 0:
			#this rank is not a root, should have >0 parent
			taxonomy[ranks[i]]["parent"].append(ranks[i-1])
		if i < len(ranks)-1:
			#this rank is not a leaf, should have >0 children
			taxonomy[ranks[i]]["children"].append(ranks[i+1])

#remove duplicate parents/children from the taxonomy
keys = taxonomy.keys()
for key in keys:
	if "children" in taxonomy[key].keys():
		children = list(set(taxonomy[key]["children"]))
		taxonomy[key]["children"] = children
	if "parent" in taxonomy[key].keys():
		parent = list(set(taxonomy[key]["parent"]))
		taxonomy[key]["parent"] = parent

#save taxonomy to file
with open("taxonomy_full.json","w") as f:
	json.dump(taxonomy, f)

#store descendants of each node in the taxonomy
keys = taxonomy.keys()
for key in keys:
	print("Key: ", key)
	if len(taxonomy[key]["children"])>0:
		descendants = get_descendants(key)
		taxonomy[key]["descendants"] = descendants
		print("Descendants = ", len(descendants))
	else:
		taxonomy[key]["descendants"] = []
		print("Descendants = 0")

#save taxonomy to file
with open("taxonomy_full.json","w") as f:
	json.dump(taxonomy, f)


#store the RNA sequences of each organism
i = 0

while i < range(546182,len(silva)):
	print("Row ", i)
	row = silva[i]
	if row[0]==">":
		accession = row[1:row.index(" ")]
		seq = ""
		i+=1
		row = silva[i]
		while row[0]!=">":
			seq+=row.strip()
			i+=1
			row = silva[i]
		if accession in silva_dict.keys():
			silva_dict[accession]["sequence"] = seq
			print("Accession: ", accession)
		else:
			print("Accession not found: ", accession)

with open("silva_dict_full.json") as f:
	json.dump(silva_dict, f)




##### RDP dataset #####

f = open("trainset14_032015.rdp.tax","r")
rdp = list(f)

def get_descendants(rank):
	descendants = []
	q = [rank]
	keys = rdp_taxonomy.keys()
	while q:
		current_term = q.pop(0)
		descendants.append(current_term)
		for key in keys:
			if current_term in rdp_taxonomy[key]["parent"]:
				if key not in descendants:
					q.append(key)
	return list(set(descendants))


rdp_dict = {}
for i in range(len(rdp)):
	print("Row ", i)
	row = rdp[i]
	row = row.split("\t")
	accession = row[0]
	ranks = row[1].strip()
	ranks = ranks.split(";")[:-1]
	for j in range(1,len(ranks)):
		ranks[j] = ranks[j-1]+" "+ranks[j]
	rdp_dict[accession] = {}
	rdp_dict[accession]["class"] = ranks

rdp_taxonomy = {}
keys = rdp_dict.keys()
for j in range(len(keys)):
	print("Organism ", j)
	key = keys[j]
	ranks = rdp_dict[key]["class"]
	for i in range(len(ranks)):
		if ranks[i] in rdp_taxonomy.keys():
			#rank already exists, update count
			rdp_taxonomy[ranks[i]]["count"] += 1
		else:
			#rank is new, create new entry for count/children/parent
			rdp_taxonomy[ranks[i]] = {}
			rdp_taxonomy[ranks[i]]["count"] = 1
			rdp_taxonomy[ranks[i]]["parent"] = []
			rdp_taxonomy[ranks[i]]["children"] = []
		if i > 0:
			#this rank is not a root, should have at least 1 parent
			rdp_taxonomy[ranks[i]]["parent"].append(ranks[i-1])
		if i < len(ranks)-1:
			#this rank is not a leaf, should have at least 1 child(ren)
			rdp_taxonomy[ranks[i]]["children"].append(ranks[i+1])

#remove duplicate parents/children from the taxonomy
keys = rdp_taxonomy.keys()
for key in keys:
	if "children" in rdp_taxonomy[key].keys():
		children = list(set(rdp_taxonomy[key]["children"]))
		rdp_taxonomy[key]["children"] = children
	if "parent" in rdp_taxonomy[key].keys():
		parent = list(set(rdp_taxonomy[key]["parent"]))
		rdp_taxonomy[key]["parent"] = parent

#store taxonomy to file
with open("rdp_taxonomy.json","w") as f:
	json.dump(rdp_taxonomy,f)

#store descendants of each node in the taxonomy
keys = rdp_taxonomy.keys()
for key in keys:
	print("Key: ", key)
	if len(rdp_taxonomy[key]["children"])>0:
		descendants = get_descendants(key)
		rdp_taxonomy[key]["descendants"] = descendants
		print("Descendants = ", len(descendants))
	else:
		rdp_taxonomy[key]["descendants"] = []
		print("Descendants = 0")

#save taxonomy to file
with open("rdp_taxonomy.json","w") as f:
	json.dump(rdp_taxonomy, f)


#store the RNA sequences of each organism
f = open("trainset14_032015.rdp.fasta", "r")
rdp = list(f)

f = open("rdp_data_dict.json","r")
rdp_dict = json.load(f)

i = 0
while i < range(21350,len(rdp)):
	print("Row ", i)
	row = rdp[i]
	if "\t" in row:
		accession = row[1:row.index("\t")]
	else:
		accession = row.split()[0][1:]
	i+=1
	seq = rdp[i].strip()
	i+=1
	if accession in rdp_dict.keys():
		rdp_dict[accession]["sequence"] = seq
		print("Accession: ", accession)
	else:
		print("Accession not found: ", accession)

with open("rdp_data_dict.json","w") as f:
	json.dump(rdp_dict, f)



