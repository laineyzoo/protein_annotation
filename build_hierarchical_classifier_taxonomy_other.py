from __future__ import division
from __future__ import print_function

import numpy as np
import collections
import json
import string
import re
import sys, getopt
from time import time

from sklearn.metrics import roc_auc_score
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


## shuffle the dataset
def shuffle_data_combined(sequence, sequence_ids, classification, actual_classification):
	sequence_shuffle = []
	sequence_ids_shuffle = []
	classes_shuffle = []
	actual_classes_shuffle = []
	index_shuffle = np.arange(len(sequence))
	np.random.shuffle(index_shuffle)
	for i in index_shuffle:
		sequence_shuffle.append(sequence[i])
		sequence_ids_shuffle.append(sequence_ids[i])
		classes_shuffle.append(classification[i])
		actual_classes_shuffle.append(actual_classification[i])
	return (sequence_shuffle, sequence_ids_shuffle, classes_shuffle, actual_classes_shuffle)

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

		other_sequence_ids = list()
		other_sequence = list()
		other_classification = list()

		print("\nPreparing the dataset")
		#from this point, all nodes not under ROOT will be considered as OTHER
		taxonomy[root]["parent"] = []
		desc_root = taxonomy[root]["descendants"]

		keys = silva_dict.keys()
		for key in keys:
			if silva_dict[key]["class"][-1] in desc_root:
				sequence_ids.append(key)
				seq = create_kmers(silva_dict[key]["sequence"],kmer)
				sequence.append(seq)
				classification.append(silva_dict[key]["class"][-1])
			else:
				other_sequence_ids.append(key)
				seq = create_kmers(silva_dict[key]["sequence"],kmer)
				other_sequence.append(seq)
				other_classification.append(silva_dict[key]["class"][-1])

		### building the binary classifier for ROOT (1) and OTHER (0)

		#truncate the OTHER data so that it is equal to ROOT data
		root_len = len(sequence_ids)
		other_sequence_ids[:root_len]
		other_sequence[:root_len]
		other_classification[:root_len]

		#shuffle the ROOT dataset
		(sequence, sequence_ids, classification) = shuffle_data(sequence,sequence_ids,classification)

		#divide the ROOT dataset
		index = int(len(sequence)/5)*4
		X_train = sequence[:index]
		sequence_ids_train = sequence_ids[:index]
		classification_train = classification[:index]
		
		X_test = sequence[index:]
		sequence_ids_test = sequence_ids[index:]
		classification_test = classification[index:]

		#shuffle the OTHER dataset
		(other_sequence, other_sequence_ids, other_classification) = shuffle_data(other_sequence,other_sequence_ids,other_classification)

		#divide the OTHER dataset
		index = int(len(other_sequence)/5)*4
		X_train_other = other_sequence[:index]
		sequence_ids_train_other = other_sequence_ids[:index]
		classification_train_other = other_classification[:index]
		
		X_test_other = other_sequence[index:]
		sequence_ids_test_other = other_sequence_ids[index:]
		classification_test_other = other_classification[index:]

		#prepare the ROOT+OTHER datasets
		X_train_combined = X_train+X_train_other
		sequence_ids_train_combined = sequence_ids_train+sequence_ids_train_other
		classification_train_combined = [1]*len(X_train)+[0]*len(X_train_other)
		actual_classes_train = classification_train+classification_train_other

		X_test_combined = X_test+X_test_other
		sequence_ids_test_combined = sequence_ids_test+sequence_ids_test_other
		classification_test_combined = [1]*len(X_test)+[0]*len(X_test_other)
		actual_classes_test = classification_test+classification_test_other

		#shuffle the ROOT+OTHER train and test sets
		(X_train_combined, sequence_ids_train_combined, classification_train_combined, actual_classes_train) = shuffle_data_combined(X_train_combined,sequence_ids_train_combined,classification_train_combined, actual_classes_train)
		(X_test_combined, sequence_ids_test_combined, classification_test_combined, actual_classes_test) = shuffle_data(X_test_combined,sequence_ids_test_combined,classification_test_combined, actual_classes_test)


		#vectorize the ROOT+OTHER dataset
		vectorizer_combined = TfidfVectorizer(sublinear_tf=True, max_df=0.5)
		X_train_combined = vectorizer_combined.fit_transform(X_train_combined)
		X_test_combined = vectorizer_combined.transform(X_test_combined)

		#create a single binary classifier (SVM or Naive-Bayes)
		if algo != "S":
			other_classifier = MultinomialNB(alpha=.01).fit(X_train_combined, classification_train_combined)
		else:
			other_classifier = svm.SVC(probability=True)
			other_classifier.fit(X_train_combined, classification_train_combined)

		print("Run the binary classifier on the test set")
		prob = other_classification.predict_proba(X_test_combined)
		y_scores = prob[:,1]
		#get ROC/AUC metrics
		roc_score = roc_auc_score(classification_test_combined, y_scores)
		print("ROC score: ", roc_score)
		#get best threshold
		precision,recall,thresholds = precision_recall_curve(classification_train_combined,y_scores)
		f1 = (2*precision*recall)/(precision+recall)
		max_thresh = thresholds[f1.argmax()]


		#build the hierarchical classifier
		vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5)
		X_train = vectorizer.fit_transform(X_train)
	
		#train hierarchical classifier
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
		print("Running hierarchical classifier on the those classified as ROOT by the binary classifier")
		prob_dict = {}
		actual_classes = {}
		for i in range(len(sequence_ids_test_combined)):
			if y_scores[i] >= max_thresh:
				test_point = vectorizer.transform(X_test_combined[i])
				prob_dict[sequence_ids_test_combined[i]] = predict_class(test_point)
				actual_classes[sequence_ids_test_combined[i]] = actual_classes_test[i]
		sequence_ids_test = prob_dict.keys()

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
				true_labels = propagate_labels([actual_classes[sequence_ids_test[i]]])
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



