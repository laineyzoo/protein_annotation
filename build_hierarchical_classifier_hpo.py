from __future__ import division
from __future__ import print_function

import numpy as np
import csv
import collections
import json
import string
import re
import sys, getopt
from time import time

from stemming.porter2 import stem
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm

#list of stopwords
f = open("english")
stopwords = list(f)
stopwords = [w.strip() for w in stopwords]

#open GO files
HPO_JSON = "hpo.json"
f = open(HPO_JSON)
hpo_ontology = json.load(f)
f.close()

HPO_PARENTS = "hpo_parents.json"
f = open(HPO_PARENTS)
hpo_parents = json.load(f)
f.close()

HPO_CHILDREN = "hpo_children.json"
f = open(HPO_CHILDREN)
hpo_children = json.load(f)
f.close()

HPO_DESC = "go_descendants.json"
f = open(HPO_DESC)
hpo_descendants = json.load(f)
f.close()

HPO_ANCES = "go_ancestors.json"
f = open(HPO_ANCES)
hpo_ancestors = json.load(f)
f.close()


def get_ancestors(hpo_term):
	return go_ancestors[hpo_term]

def get_descendants(hpo_term):
	return hpo_descendants[hpo_term]

def get_children(hpo_term):
	return hpo_children[hpo_term]

def get_parents(hpo_term):
	return hpo_parents[hpo_term]


## randomize the ordering of the dataset
def shuffle_data(abstracts, hpo_terms, pmids):
	print("Shuffle dataset")
	abstracts_shuffle = []
	hpo_terms_shuffle = []
	pmids_shuffle = []
	index_shuffle = np.arange(len(abstracts))
	np.random.shuffle(index_shuffle)
	for i in index_shuffle:
		abstracts_shuffle.append(abstracts[i])
		hpo_terms_shuffle.append(hpo_terms[i])
		pmids_shuffle.append(pmids[i])
	return (abstracts_shuffle, hpo_terms_shuffle, pmids_shuffle)



def text_preprocessing(text):
    #lowercase everything
    text = text.lower()
    #replace punctuation with single space
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    text = regex.sub(" ", text)
    #remove stopwords
    no_stopwords = [word for word in text.split() if word.lower() not in stopwords]
    text = " ".join(no_stopwords)
    #stem the words
    text = " ".join([stem(w) for w in text.split()])
    return text


#compute precision & recall for one test point
def evaluate_prediction(true_labels, pred_labels):
	inter = set(pred_labels) & set(true_labels)
	precision = 0
	recall = 0
	if len(pred_labels)!=0:
		precision = len(inter)/len(pred_labels)
	if len(true_labels)!=0:
		recall = len(inter)/len(true_labels)
	return precision,recall


#get set of ancestors for each go term
def propagate_hpo_terms(hpo_terms):
	label_list = []
	for term in hpo_terms:
		label_list.extend(get_ancestors(term))
	return list(set(label_list))


def predict_terms(test_point):
	prob_ontology = []
	for clf in classifiers:
		prob = clf.predict_proba(test_point)[0][1]
		prob_ontology.append(prob)
	return prob_ontology




####################################### MAIN #############################################
if __name__ == "__main__":
	
	if len(sys.argv[1:]) < 1:
		print("This script requires at least 4 arguments: sample_threshold, save_results")
		exit()
	else:
		print("=====START=====")
		print("\nSettings:")

		sample_threshold = int(sys.argv[1])
		print("Sample threshold: ", sample_threshold)
	
		save_results = sys.argv[2]
		print("Save results? ", save_results)

		time_start_all = time()

		#prepare train set
		fname1 = "pubmed_hpo_papers_dict.json"
		fname2 = "pubmed_hpo_papers.json"

		f = open(fname1,"r")
		hpo_papers_dict = json.load(f)
		f.close()

		f = open(fname2,"r")
		pubmed_papers_dict = json.load(f)
		f.close()

		hpo_papers_keys = hpo_papers_dict.keys()
		pubmed_papers_keys = pubmed_papers_dict.keys()
		for node in hpo_ontology:
			hpo_term = node['id']
			if hpo_term in hpo_papers_keys:
				pubmed_ids = hpo_papers_dict[hpo_term]
				if len(pubmed_ids)>10:
					pubmed_ids = pubmed_ids[:10]
				for pmid in pubmed_ids:
					if pmid in pubmed_papers_keys:
						text = pubmed_papers_dict[pmid]
						text = text_preprocessing(text)
						data_list.append(text)
						class_list.append(hpo_term)
						id_list.append(pmid)

		#shuffle train set
		(data_list, class_list, id_list) = shuffle_data(data_list, class_list, id_list)

		fraction_train_data = 1
		index = int(len(data_list)/fraction_train_data)
		X_train = data_list[:index]
		class_train = class_list[:index]
		id_train = id_list[:index]

		#prepare test set
		f = open("hpo_gold_dict.json")
		hpo_gold = json.load(f)
		f = open("pubmed_records.json")
		pubmed_records = json.load(f)
	
		X_test = []
		class_test = []
		id_test = []
		
		hpo_prot = hpo_gold.keys()
		for prot in hpo_prot:
			pmids = hpo_gold[prot]["PMID"]
			hpo_terms = hpo_gold[prot]["HPO"]
			text = ""
			for pmid in pmids:
				if pmid in pubmed_records.keys():
					text+=pubmed_records[pmid]
			text = text_preprocessing(text)
			X_test.append(text)
			class_test.append(hpo_terms)
			id_test.append(prot)

		train_len = len(X_train)
		test_len = len(X_test)	
		print("Train set: ", train_len)
		print("Test set: ", test_len)
		
		#vectorize features
		vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5)
		X_train = vectorizer.fit_transform(X_train)
		X_test = vectorizer.transform(X_test)
		
		#create binary classifiers
		print("Creating classifiers")
		time_start_classifier = time()
		classifiers = []
		classifier_keys = []
		for node in hpo_ontology:
			hpo_id = node['id']
			descendants = get_descendants(hpo_id)
			y_train = []
			for term in class_train:
				if term in descendants:
					y_train.append(1)
				else:
					y_train.append(0)
			pos_count = y_train.count(1)
			if pos_count>=sample_threshold:
				clf = MultinomialNB().fit(X_train, y_train)
				classifiers.append(clf)
				classifier_keys.append(hpo_id)
		print("Done creating classifiers. Classifier count: ", len(classifiers))
		time_end_classifier = time()-time_start_classifier


		print("Running the classifiers on the test set")
		time_start_test = time()
		prob_dict = {}
		for i in range(len(id_test)):
			test_point = X_test[i]
			prob_dict[id_test[i]] = predict_terms(test_point)
		time_end_test = time()-time_start_test
	
		print("Calculate F1/recall/precision by threshold")
		time_start_eval = time()
		precision_list = []
		recall_list = []
		f1_list = []
		for r in range(1,101):
			thresh = r/100
			total_precision = 0
			total_recall = 0
			for i in range(test_len):
				all_labels = np.array(classifier_keys)
				all_prob = np.array(prob_dict[id_test[i]])
				positive_labels = list(all_labels[all_prob[:]>=thresh])
				predicted_labels = propagate_hpo_terms(positive_labels)
				true_labels = propagate_hpo_terms(class_test[i])
				precision,recall = evaluate_prediction(true_labels, predicted_labels)
				total_precision+=precision
				total_recall+=recall
			final_precision = total_precision/test_len
			final_recall = total_recall/test_len
			final_f1 = 0
			if final_precision+final_recall>0:
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
		print("Maximizing Threshold: ", max_thresh)
		

		print("\n-----Timings-----")
		print("Total time: ", time_end_all)

		#save output to file
		f = open("log_HPO.txt","w")
		print("\n-----Results-----", file=f)
		print("Max F1: ", max_f1, file=f)
		print("Max Precision: ", max_precision, file=f)
		print("Max Recall: ", max_recall, file=f)
		print("Max Threshold: ", max_thresh, file=f)

		print("\n-----Timings-----", file=f)
		print("Creating classifiers: ", time_end_classifier, file=f)
		print("Evaluating test set: ", time_end_test, file=f)
		print("Computing metrics per threshold: ", time_end_eval, file=f)
		print("Total time: ", time_end_all, file=f)

		print("\nDONE!")





