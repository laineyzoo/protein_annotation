from __future__ import division
from __future__ import print_function

import numpy as np
import collections
import json
import string
import re
import sys, getopt
from time import time
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm



## shuffle the dataset
def shuffle_data(sequence, sequence_ids, classification):
	sequence_shuffle = []
	sequence_ids_shuffle = []
	classes_shuffle = []
	index_shuffle = np.arange(len(sequence))
	np.random.shuffle(index_shuffle)
	for i in index_shuffle:
		sequence_shuffle.append(sequence[i])
		sequence_ids_shuffle.append(sequence_ids[i])
		classes_shuffle.append(classification[i])
	return (sequence_shuffle, sequence_ids_shuffle, classes_shuffle)


#divide a sequence into kmers
def create_kmers(sequence, k):
	if k<len(sequence):
		kmers = list()
		for i in range(len(sequence)+1-k):
			kmers.append(sequence[i:i+k])
		seq = " ".join(k for k in kmers)
		return seq
	else:
		return sequence

#check if all elements of an array are equal
def array_equal(arr):
	return all(np.isclose(x, arr[0]) for x in arr)

#get distance to root
def path_to_root(rank, taxonomy):
	if root==rank:
		return 0
	q = collections.deque()
	q.append(rank)
	dist = 0
	taxonomy[root]["parent"] = []
	while q:
		node = q.popleft()
		dist += 1
		parents = taxonomy[node]["parent"]
		q.extend(parents)
	return (dist-1)


#return LCA of a list of nodes
def get_lca(node_list, taxonomy):
	if len(node_list)==0:
		return ""
	elif len(node_list)==1:
		return node_list[0]
	paths = list()
	[paths.append(propagate_labels([node], taxonomy)) for node in node_list]
	intersect = list()
	path0 = paths[0]
	path1 = paths[1]
	intersect = list(set(path0) & set(paths1))
	if len(node_list)>2:
		for i in range(2, len(paths)):
			intersect = list(set(intersect) & set(paths[i]))
	intersect_len = [path_to_root(inter, taxonomy) for inter in intersect]
	lca = intersect[intersect_len.index(max(intersect_len))]
	return lca


def filter_prediction(pred_labels, idx):
	pred = pred_labels
	path_len = [path_to_root(j, taxonomy) for j in pred]
	counts = collections.Counter(path_len)
	values = counts.values()
	error_type = 0
	#check if we have an ambiguous prediction
	if max(values)>1:
		lowest_pred_len = max(path_len)
		prob_ont = prob_dict[sequence_ids_test[idx]]
		#check if Case 1 vs 2 & 3
		if counts[lowest_pred_len]>1:
			lowest_labels = list()
			lowest_labels_prob = list()
			for i in range(len(path_len)):
				if path_len[i]==lowest_pred_len:
					lowest_labels.append(pred[i])
					lowest_labels_prob.append(prob_ont[pred[i]])
			#check for Case 3: ambiguous prediction is at the lowest level and 1 or more higher levels
			if sum((np.array(values)>1)*1)>1:
				print("Case 3")
				error_type = 3
				if array_equal(lowest_labels_prob):
					pred = filter_prediction(lowest_labels, idx)
				else:
					highest_prob_label = lowest_labels[lowest_labels_prob.index(max(lowest_labels_prob))]
					pred = propagate_labels([highest_prob_label], taxonomy)
			else:
				#Case 2: ambiguous prediction is at the lowest level only
				print("Case 2")
				error_type = 2
				#Case 2 solution (check for Case 4)
				#if all prob are equal, return LCA
				if array_equal(lowest_labels_prob):
					print("All sibling prob are equal. Get LCA.")
					#we just need to return LCA since it will just get propagated upwards later
					lca = get_lca(lowest_labels, taxonomy)
					pred = propagate_labels([lca], taxonomy)
				#one prob is higher than the rest, remove the rest
				else:
					print("One sibling has higher prob, return it.")
					highest_prob_label = lowest_labels[lowest_labels_prob.index(max(lowest_labels_prob))]
					pred = propagate_labels([highest_prob_label], taxonomy)
		#Case 1
		else:
			#Case 1: ambiguous prediction is at a higher level but not in the lowest level
			print("Case 1 ")
			error_type = 1
			max_val = max(values)
			ambiguous_level = 0
			for k in counts.keys():
				if counts[k]==max_val:
					ambiguous_level = k
			ambiguous_labels = list()
			ambiguous_labels_prob = list()
			for i in range(len(path_len)):
				if path_len[i]==ambiguous_level:
					ambiguous_labels.append(pred[i])
					ambiguous_labels_prob.append(prob_ont[pred[i]])
			print("Ambiguous level: ", ambiguous_level)
			lowest_label = pred_labels[path_len.index(lowest_pred_len)]
			longest_path = propagate_labels([lowest_label], taxonomy)
			highest_prob_label = ambiguous_labels[ambiguous_labels_prob.index(max(ambiguous_labels_prob))]
			#check if probs at the ambiguous level are equal
			if array_equal(ambiguous_labels_prob) or (highest_prob_label in longest_path):
				print("All sib prob are equal | highest prob sib is in longest path, return longest path.")
				pred = longest_path
			else:
				#one sibling has higher prob, return it
				print("One sibling has higher prob and not in the longest path")
				pred = propagate_labels([highest_prob_label], taxonomy)
	else:
		print("No ambiguous predictions. Do nothing.")
	return pred




#compute precision & recall for one test point
def evaluate_prediction(true_labels, pred_labels):
	inter = set(pred_labels).intersection(true_labels)
	precision = 0
	recall = 0
	if len(pred_labels)!=0:
		precision = len(inter)/len(pred_labels)
	if len(true_labels)!=0:
		recall = len(inter)/len(true_labels)
	return precision,recall


# propagate the given label/rank upwards in the taxonomy until it reaches a root
def propagate_labels(labels,taxonomy):
	label_list = []
	for label in labels:
		labels_prop = list()
		labels_prop.append(label)
		q = collections.deque()
		q.append(label)
		#traverse ontology upwards from node to root
		taxonomy[root]["parent"] = []
		while len(q)>0:
			node = q.popleft()
			parents = taxonomy[node]["parent"]
			labels_prop.extend(parents)
			q.extend(parents)
		#remove duplicates in the label set
		labels_prop = list(set(labels_prop))
		#add this label set to our list
		label_list.extend(labels_prop)
	label_list = list(set(label_list))
	return label_list


#give prediction probabilities for this test point
def predict_class(test_point):
	prob_ontology = {}
	for key in classifiers.keys():
		clf = classifiers[key]
		prob = clf.predict_proba(test_point)[0]
		classes = clf.classes_
		if len(classes)==2:
			positive_prob = prob[classes[:]==1]
		else:
			if classes[0]==0:
				positive_prob = 0.0
			else:
				positive_prob = 1.0
		prob_ontology[key] = positive_prob
	return prob_ontology



######################################  MAIN  ############################################


if __name__ == "__main__":
	if len(sys.argv[1:]) < 4:
		print("This script requires 6 arguments: root, kmer size, sample_threshold, classifier, dataset, filtering")
	else:
		time_start_all = time()
		print("\nSTART\n")
		print("Settings: ")

		root = sys.argv[1]
		print("Root: ", root)

		kmer = int(sys.argv[2])
		print("K-mer size: ", kmer)
	
		sample_threshold = int(sys.argv[3])
		print("Sample threshold: ", sample_threshold)
	
		algo = sys.argv[4]
		if algo == "S":
			print("Classifier: SVM")
		else:
			print("Classifier: Naive-Bayes")
		
		dataset = sys.argv[5]
		print("dataset: ",dataset)
		if dataset == "R":
			print("Dataset: RDP")
		else:
			print("Dataset: SILVA")
		
		filter = sys.argv[6]
		print("Filter: ", filter)
		
		#from this point, all nodes above the root will not be considered
		if dataset == "R":
			f1 = open("rdp_taxonomy.json","r")
			f2 = open("rdp_data_dict.json", "r")
		else:
			f1 = open("taxonomy.json", "r")
			f2 = open("silva_dict.json", "r")

		taxonomy = json.load(f1)
		data_dict = json.load(f2)

		#Only use the parts of the dataset that belongs to the root
		desc_root = taxonomy[root]["descendants"]
		print("Descendants: ", len(desc_root))

sequence_ids = list()
sequence = list()
classification = list()

keys = data_dict.keys()
for key in keys:
		if data_dict[key]["class"][-1] in desc_root:
				sequence_ids.append(key)
				seq = create_kmers(data_dict[key]["sequence"],kmer)
				sequence.append(seq)
				classification.append(data_dict[key]["class"][-1])


#shuffle dataset
print("\nPreparing the dataset")
(sequence, sequence_ids, classification) = shuffle_data(sequence, sequence_ids, classification)

#divide dataset
index = int(len(sequence_ids)/5)*4

X_train = sequence[:index]
sequence_ids_train = sequence_ids[:index]
classification_train = classification[:index]

X_test = sequence[index:]
sequence_ids_test = sequence_ids[index:]
classification_test = classification[index:]

		print("Train set: ", len(X_train))
		print("Test set: ", len(X_test))
	
		print("Vectorizing dataset")
		#vectorize features
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5)
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

		#create binary classifiers
		print("Creating classifiers for taxonomy at root: ", root)
time_start_classifier = time()
classifiers = {}
positive_count = {}
for desc in desc_root:
	descendants = taxonomy[desc]["descendants"]
	y_train = list()
	for c in classification_train:
		if c in descendants:
			y_train.append(1)
		else:
			y_train.append(0)
	pos_count = y_train.count(1)
	if pos_count>=sample_threshold:
		if algo != "S" or pos_count==len(y_train):
			clf = MultinomialNB(alpha=.01).fit(X_train, y_train)
			classifiers[desc] = clf
		else:
			clf = svm.SVC(probability=True)
			clf.fit(X_train,y_train)
			classifiers[desc] = clf
		positive_count[desc] = pos_count
		print("Done creating classifiers. No. of classifiers: ", len(classifiers))
		time_end_classifier = time()-time_start_classifier

		#run classifiers on the test set
print("Running classifiers on the test set")
time_start_test = time()
prob_dict = {}
for i in range(len(sequence_ids_test)):
	test_point = X_test[i]
	prob_dict[sequence_ids_test[i]] = predict_class(test_point)
time_end_test = time()-time_start_test
	
		#plot the precision/recall/f1 vs threshold
		print("Calculating F1/recall/precision by threshold")
time_start_eval = time()
precision_list = list()
recall_list = list()
f1_list = list()
for thresh in range(98, 101):
	thresh = float(thresh)/100
	total_precision = 0
	total_recall = 0
	for i in range(len(sequence_ids_test)):
		true_labels = propagate_labels([classification_test[i]],taxonomy)
		prob_ontology = prob_dict[sequence_ids_test[i]]
		positive_labels = list()
		for key in classifiers.keys():
			if prob_ontology[key] >= thresh:
				positive_labels.append(key)
		predicted_labels = propagate_labels(positive_labels,taxonomy)
		precision,recall = evaluate_prediction(true_labels, predicted_labels)
		total_precision+=precision
		total_recall+=recall
	final_precision = total_precision/len(sequence_ids_test)
	final_recall = total_recall/len(sequence_ids_test)
	final_f1 = (2*final_precision*final_recall)/(final_precision+final_recall)
	precision_list.append(final_precision)
	recall_list.append(final_recall)
	f1_list.append(final_f1)

max_f1 = max(f1_list)
max_thresh = f1_list.index(max_f1)
max_precision = precision_list[max_thresh]
max_recall = recall_list[max_thresh]
time_end_eval = time()-time_start_eval
time_end_all = time()-time_start_all
	

		print("\n-----Results-----")
		print("Max F1: ", max_f1)
		print("Max Precision: ", max_precision)
		print("Max Recall: ", max_recall)
		print("Max Threshold: ", max_thresh)
		
		print("\n-----Timings-----")
		print("Creating classifiers: ", time_end_classifier)
		print("Evaluating test set: ", time_end_test)
		print("Computing metrics per threshold: ", time_end_eval)
		print("Total time: ", time_end_all)
		
		print("\n-----Settings-----")
		print("Root: ", root)
		print("K-mer size: ", kmer)
		print("Sample Threshold: ", sample_threshold)
		if algo == "S":
			print("Classifier Type: SVM")
		else:
			print("Classifier Type: Naive-Bayes")
		if dataset == "R":
			print("Dataset: RDP")
		else:
			print("Dataset: SILVA")
		print("Classifiers: ", len(classifiers))
		print("\nDONE!\n")


		f = open("log_"+str(time()), "w")
		print("\n-----Results-----", file=f)
		print("Max F1: ", max_f1, file=f)
		print("Max Precision: ", max_precision,file=f)
		print("Max Recall: ", max_recall, file=f)
		print("Max Threshold: ", max_thresh, file=f)

		print("\n-----Timings-----", file=f)
		print("Creating classifiers: ", time_end_classifier, file=f)
		print("Evaluating test set: ", time_end_test, file=f)
		print("Computing metrics per threshold: ", time_end_eval, file=f)
		print("Total time: ", time_end_all, file=f)

		print("\n-----Settings-----",file=f)
		print("Root: ", root, file=f)
		print("K-mer size: ", kmer, file=f)
		print("Sample Threshold: ", sample_threshold, file=f)
		if algo == "S":
				print("Classifier Type: SVM", file=f)
		else:
				print("Classifier Type: Naive-Bayes", file=f)
		if dataset == "R":
			print("Dataset: RDP", file=f)
		else:
			print("Dataset: SILVA", file=f)
		print("Classifiers: ", len(classifiers),file=f)
		print("\nDONE!\n", file=f)





true_labels = []
pred_labels = []
for i in range(len(sequence_ids_test)):
	true_labels.append(propagate_labels([classification_test[i]],taxonomy))
	prob_ontology = prob_dict[sequence_ids_test[i]]
	pos_labels = list()
	for key in classifiers.keys():
		if prob_ontology[key] >= 0.99:
			pos_labels.append(key)
	pred_labels.append(pos_labels)


for i in range(80,90):
	if len(set(pred_labels[i])-set(true_labels[i]))>0:
		print("\n", i)
		print("True Labels:", len(true_labels[i]))
		for j in range(len(true_labels[i])):
			print(true_labels[i][j])
		print("\nPred Labels:", len(pred_labels[i]))
		for j in range(len(pred_labels[i])):
			print(pred_labels[i][j])


f = open("taxonomy_latest.json","r")
taxonomy = json.load(f)
f = open("silva_dict_latest.json","r")
silva_dict = json.load(f)

keys = taxonomy.keys()
unid = list()
for k in keys:
	if "unidentified" in k:
		unid.append(k)

for k in keys:
	if k in unid:
		del taxonomy[k]

keys = taxonomy.keys()
for k in keys:
	children = taxonomy[k]["children"]
	descendants = taxonomy[k]["descendants"]
	[children.remove(c) for c in children if c in unid]
	[descendants.remove(d) for d in descendants if d in unid]
	taxonomy[k]["children"] = children
	taxonomy[k]["descendants"] = descendants

silva_keys = silva_dict.keys()
for k in silva_keys:
	species = silva_dict[k]["class"][-1]
	if species in unid:
		del silva_dict[k]


unclass = list()

for k in keys:
	if "unclassified" in k:
		unclass.append(k)

for k in keys:
	if k in unclass:
		del taxonomy[k]

keys = taxonomy.keys()
for k in keys:
	children = taxonomy[k]["children"]
	descendants = taxonomy[k]["descendants"]
	[children.remove(c) for c in children if c in unclass]
	[descendants.remove(d) for d in descendants if d in unclass]
	taxonomy[k]["children"] = children
	taxonomy[k]["descendants"] = descendants

silva_keys = silva_dict.keys()
for k in silva_keys:
	species = silva_dict[k]["class"][-1]
	if species in unclass:
		del silva_dict[k]

meta = list()
for k in keys:
	if "metagenome" in k or "metagenomic" in k:
		meta.append(k)

for k in keys:
	if k in meta:
		del taxonomy[k]

keys = taxonomy.keys()
for k in keys:
	children = taxonomy[k]["children"]
	descendants = taxonomy[k]["descendants"]
	[children.remove(c) for c in children if c in meta]
	[descendants.remove(d) for d in descendants if d in meta]
	taxonomy[k]["children"] = children
	taxonomy[k]["descendants"] = descendants

silva_keys = silva_dict.keys()
for k in silva_keys:
	species = silva_dict[k]["class"][-1]
	if species in meta:
		del silva_dict[k]


environ = list()
for k in keys:
	if "environmental" in k:
		environ.append(k)

keys = taxonomy.keys()
for k in keys:
	if k in environ:
		del taxonomy[k]

keys = taxonomy.keys()
for k in keys:
	children = taxonomy[k]["children"]
	descendants = taxonomy[k]["descendants"]
	[children.remove(c) for c in children if c in environ]
	[descendants.remove(d) for d in descendants if d in environ]
	taxonomy[k]["children"] = children
	taxonomy[k]["descendants"] = descendants

silva_keys = silva_dict.keys()
for k in silva_keys:
	species = silva_dict[k]["class"][-1]
	if species in environ:
		del silva_dict[k]

tax_keys = taxonomy.keys()
silva_keys = silva_dict.keys()
not_leaf = 0
for k in silva_keys:
	species = silva_dict[k]["class"][-1]
	if species not in tax_keys:
		del silva_dict[k]
	elif len(taxonomy[species]["children"])>0:
		print("Not a leaf, delete sample")
		not_leaf+=1
		del silva_dict[k]


tax_keys = rdp_taxonomy.keys()
rdp_keys = rdp_dict.keys()
not_leaf = 0
for k in rdp_keys:
	genus = rdp_dict[k]["class"][-1]
	if genus not in tax_keys:
		print("Genus not in taxonomy")
		del rdp_dict[k]
	elif len(rdp_taxonomy[genus]["children"])>0:
		print("Not a leaf, delete sample")
		not_leaf+=1
		del rdp_dict[k]
