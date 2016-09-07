from __future__ import division
from __future__ import print_function

import numpy as np
import json
import string


f = open("SILVA_123.1_SSURef_Nr99_tax_silva_trunc.fasta", "r")
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
	silva_dict[accession] = ranks


taxonomy = {}
keys = silva_dict.keys()
for i in range(len(keys)):
	print("Organism ", i)
	key = keys[i]
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

#check for cycles by topologically sorting the tree
q = collections.deque()
q.append("None")

count = 0
while len(q)>0:
	id = q.popleft()
	count+=1
	print("Count: ", count)
	if "children" in taxonomy[id].keys():
		children = taxonomy[id]["children"]
		for child in children:
			q.append(child)


#store the RNA sequences of each organism
silva_seq_dict = {}
i = 0
while i < range(len(silva)):
	row = silva[i]
	if row[0]==">":
		name = row[1:row.index(" ")]
		seq = ""
		i+=1
		row = silva[i]
		while row[0]!=">":
			seq+=row.strip()
			i+=1
			row = silva[i]
		silva_dict[name]["sequence"] = seq
		print("Name: ", name)
		print("Organisms added: ", len(silva_seq_dict))


found = 0
not_found = []
silva_dict_2 = {}
keys = silva_dict.keys()
keys_seq = silva_seq_dict.keys()
for k in keys:
	k2 = k[:len(k)-1]
	if k2 in keys_seq:
		silva_dict_2[k] = {}
		silva_dict_2[k]["class"] = silva_dict[k]
		silva_dict_2[k]["sequence"] = silva_seq_dict[k2]
		found+=1
		print("found = ", found)
	else:
		print("not found")
		not_found.append(k)