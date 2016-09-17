from __future__ import division
from __future__ import print_function

import numpy as np
import collections
import json
import string
import re
import sys, getopt
from time import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm


with open("taxonomy.json","r") as f:
	taxonomy = json.load(f)
	taxonomy_keys = taxonomy.keys()
with open("silva_dict.json","r") as f:
	silva_dict = json.load(f)
	silva_dict_keys = silva_dict.keys()


#return all descendants of the named rank from the taxonomy
def get_descendants(rank):
	descendants = taxonomy[rank]["descendants"]
	return list(set(descendants))


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


# propagate the given label/rank upwards in the taxonomy until it reaches the designated root
def propagate_labels(labels):
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

#get distance to root
def path_to_root(rank):
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
		print("This script requires 4 arguments: root, kmer size, sample_threshold, classifier")
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
		
		sequence_ids = list()
		sequence = list()
		classification = list()

		#from this point, all nodes above the root will not be considered
		taxonomy[root]["parent"] = []

		#Only use the parts of the dataset that belongs to the root
		desc_root = taxonomy[root]["descendants"]

		keys = silva_dict.keys()
		for key in keys:
			if silva_dict[key]["class"][-1] in desc_root:
				sequence_ids.append(key)
				seq = create_kmers(silva_dict[key]["sequence"],kmer)
				sequence.append(seq)
				classification.append(silva_dict[key]["class"][-1])

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

		#vectorize features
		vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5)
		X_train = vectorizer.fit_transform(X_train)
		X_test = vectorizer.transform(X_test)

		#create binary classifiers
		print("Creating classifiers for taxonomy at root: ", root)
		node_count = 0
		classifiers = {}
		positive_count = {}
		for desc in desc_root:
			descendants = taxonomy[desc]["descendants"]
			node_count+=1
			y_train = list()
			for c in classification_train:
				if c in descendants:
					y_train.append(1)
				else:
					y_train.append(0)
			pos_count = y_train.count(1)
			if pos_count>=sample_threshold:
				if algo != "S" or y_train.count(1)==len(y_train):
					clf = MultinomialNB(alpha=.01).fit(X_train, y_train)
				else:
					clf = svm.SVC(probability=True)
					clf.fit(X_train,y_train)
				classifiers[desc] = clf
				positive_count[desc] = pos_count
		print("Done creating classifiers. No. of classifiers: ", len(classifiers))

		#run classifiers on the test set
		print("Running classifiers on the test set")
		prob_dict = {}
		for i in range(len(sequence_ids_test)):
			test_point = X_test[i]
			prob_dict[sequence_ids_test[i]] = predict_class(test_point)

				#plot the precision/recall/f1 vs threshold
		print("Calculating F1/recall/precision by threshold")
		precision_list = list()
		recall_list = list()
		f1_list = list()
		for thresh in range(0,101):
			thresh = float(thresh)/100
			print("threshold = ", thresh)
			total_precision = 0
			total_recall = 0
			for i in range(len(sequence_ids_test)):
				true_labels = propagate_labels([classification_test[i]])
				prob_ontology = prob_dict[sequence_ids_test[i]]
				positive_labels = list()
				for key in classifiers.keys():
					if prob_ontology[key] >= thresh:
						positive_labels.append(key)
				predicted_labels = propagate_labels(positive_labels)
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
		print("Total time: ", time()-time_start_all)
		
		print("\n-----Settings-----")
		print("Root: ", root)
		print("K-mer size: ", kmer)
		print("Sample Threshold: ", sample_threshold)
		if algo == "S":
			print("Classifier Type: SVM")
		else:
			print("Classifier Type: Naive-Bayes")
		print("\nDONE!\n")



