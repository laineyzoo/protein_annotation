from __future__ import division
from __future__ import print_function

import numpy as np
import collections
import json
import string
import re
import sys, getopt
import pickle
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
		print("This script requires 5 arguments: root, kmer size, sample_threshold, classifier, dataset")
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
		if dataset == "R":
			print("Dataset: RDP")
		else:
			print("Dataset: SILVA")

		sequence_ids = list()
		sequence = list()
		classification = list()
		
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
		results_dict = {}
		time_start_eval = time()
		precision_list = list()
		recall_list = list()
		f1_list = list()
		for thresh in range(0, 101):
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
				results_dict[sequence_ids_test[i]] = {}
				results_dict[sequence_ids_test[i]]["true"]=true_labels
				results_dict[sequence_ids_test[i]]["predicted"]=predicted_labels
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
		print("Script: ", sys.argv[0])
                print("Taxonomy data: ", f1)
                print("Sequence data: ", f2)
		print("Saving results_dict to file...")
		with open("results/results_"+root+"_"+str(kmer)+"_"+str(sample_threshold)+"_"+algo+"_"+dataset+".json","w") as f:
			json.dump(results_dict, f)
			print("OK!")
		#print("Saving vectorizer to file...")
		#with open("classifiers/vectorizer_"+root+"_"+str(kmer)+"_"+str(sample_threshold)+"_"+algo+"_"+dataset,"w") as f:
		#	pickle.dump(vectorizer, f)
		#	print("OK!")
		#print("Saving classifiers to file...")
		#keys = classifiers.keys()
		#for key in keys:
		#	with open("classifiers/classifiers_"+key+"_"+root+"_"+str(kmer)+"_"+str(sample_threshold)+"_"+algo+"_"+dataset,"w") as f:
		#		pickle.dump(classifiers[key], f)
		#print("OK!")
		print("DONE!\n")


		f = open("log_"+str(time()), "w")
		print("Saving performance stats...")
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
		print("Script: ", sys.argv[0],file=f)
		print("Taxonomy data: ", f1, file=f)
		print("Sequence data: ", f2, file=f)
		print("\nDONE!")

abridge_keys = []
for ab in abridge:
	idx = ab.index(".")
	key = ab[:idx]
	abridge_keys.append(key)

silva_short_keys = []
for key in silva_keys:
	idx = key.index(".")
	short_key = key[:idx]
	silva_short_keys.append(short_key)

silva_dict_abridge = {}
for ab in abridge_keys:
	if ab in silva_short_keys:
		idx = silva_short_keys.index(ab)
		silva_dict_abridge[silva_keys[idx]] = silva[silva_keys[idx]]


