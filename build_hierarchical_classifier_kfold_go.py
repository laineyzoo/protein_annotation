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

#from nltk.corpus import stopwords
#from nltk import PorterStemmer
from stemming.porter2 import stem

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm

#list of stopwords
f = open("english")
stopwords = list(f)
stopwords = [w.strip() for w in stopwords]


#list of excluded go terms
f = open("excluded_go_terms")
l = list(f)
exclude_classes = [i.strip() for i in l]


#open GO files
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

GO_DESC = "go_descendants.json"
f = open(GO_DESC)
go_descendants = json.load(f)
f.close()

GO_ANCES = "go_ancestors.json"
f = open(GO_ANCES)
go_ancestors = json.load(f)
f.close()


def get_ancestors(go_term):
	return go_ancestors[go_term]

def get_descendants(go_term):
	return go_descendants[go_term]

def get_children(go_term):
	return go_children[go_term]

def get_parents(go_term):
	return go_parents[go_term]

## randomize the ordering of the dataset ##
def shuffle_data(abstracts, go_terms, pmids):
	print("Shuffle dataset")
	abstracts_shuffle = []
	go_terms_shuffle = []
	pmids_shuffle = []
	index_shuffle = np.arange(len(abstracts))
	np.random.shuffle(index_shuffle)
	for i in index_shuffle:
		abstracts_shuffle.append(abstracts[i])
		go_terms_shuffle.append(go_terms[i])
		pmids_shuffle.append(pmids[i])
	return (abstracts_shuffle, go_terms_shuffle, pmids_shuffle)


# remove duplicate papers/proteins appearing in both train and test
def remove_duplicate_papers(pmids_train, X_test, go_terms_test, pmids_test, dataset):
	print("Remove duplicates")
	pmids_test_array = np.array(pmids_test)
	# papers in the train set should not be in the test set
	delete_indexes = list()
	for test_point in pmids_test:
		#check if this paper from the test set appears in the train set
		if test_point in pmids_train:
			#print("Test paper appears in train set")
			indexes = list(np.where(pmids_test_array==test_point)[0])
			delete_indexes.extend(indexes)
	delete_indexes = list(set(delete_indexes))
	print("Papers to delete: ", len(delete_indexes))
	for loc in sorted(delete_indexes, reverse=True):
		del X_test[loc]
		del go_terms_test[loc]
		del pmids_test[loc]
	return X_test, go_terms_test, pmids_test



def text_preprocessing(text):
    #lowercase everything
    text = text.lower()
    #remove punctuation
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    text = regex.sub(" ", text)
    #remove stopwords
    no_stopwords = [word for word in text.split() if word not in stopwords]
    text = " ".join(no_stopwords)
    #stem the words
    text = " ".join([stem(w) for w in text.split()])
    return text


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


#get set of ancestors for each go term
def propagate_go_terms(go_terms):
	label_list = []
	for term in go_terms:
		label_list.extend(get_ancestors(term))
	return list(set(label_list))


#get posterior probabilities for this test point
def predict_go(test_point):
	prob_ontology = []
	for clf in classifiers:
		prob = clf.predict_proba(test_point)[0][1]
		prob_ontology.append(prob)
	return prob_ontology

#create binary classifiers for the ontology
def create_classifiers(sample_threshold, namespace):
	classifiers = []
	classifier_keys = []
	for node in go_ontology:
		go_id = node['id']
		if node['namespace'] == namespace:
			descendants = get_descendants(go_id)
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
			classifier_keys.append(go_id)
	return classifiers, classifier_keys



####################################### MAIN #############################################
if __name__ == "__main__":
	
	if len(sys.argv[1:]) < 4:
		print("This script requires at least 4 arguments: ontology, dataset, sample threshold, no. of folds")
		exit()
	else:
		time_start_all = time()
		print("\n=======START=======")
		print("\nSettings:")
		ont = sys.argv[1]
		namespace = "cellular_component"
		if ont == "F":
			namespace = "molecular_function"
		elif ont == "P":
			namespace = "biological_process"
		print("Ontology: ", namespace)

		dataset = sys.argv[2]
		if dataset=="U":
			print("Train set: Uniprot abstracts")
			print("Test set: Uniprot abstracts")
		elif dataset == "P3":
			print("Train set: PubMed papers w/ GO names AND Uniprot")
			print("Test set: All Uniprot")
		elif dataset == "P4":
			print("Train set: PubMed papers w/ gene names AND Uniprot")
			print("Test set: All Uniprot")
		else:
			print("Invalid dataset. Valid datasets are U, P3 and P4. Exiting...")
			exit()

	
		sample_threshold = int(sys.argv[3])
		print("Sample threshold: ", sample_threshold)		
		
		folds = int(sys.argv[4])
		print("Folds: ", folds)

		#open the dataset files and create the dataset
		print("\nPreparing the dataset")

		f = open("protein_records.json","r")
		data = np.array(json.load(f))
		f.close()
		data = data[data[:,4]==ont]
		if ont == "F":
			data = data[data[:,1]!="GO:0005515"]
		unique_proteins = list(set(data[:,0]))
		uniprot_go_terms = list(set(data[:,1]))
		uniprot_pmids = list(set(data[:,2]))
		
		f = open("pubmed_records.json","r")
		data2 = json.load(f)

		allowed_go_terms = go_parents.keys()

		data_list = list()
		class_list = list()
		id_list = list()

		#dataset: UniProt abstracts
		for pmid in uniprot_pmids:
			text = text_preprocessing(data2[pmid])
			matching_proteins = data[data[:,2]==pmid]
			protein_ids = list(set(matching_proteins[:,0]))
			go_terms_protein = list(set(matching_proteins[:,1]))
			for term in go_terms_protein:
				if term in allowed_go_terms:
					data_list.append(text)
					class_list.append(term)
					id_list.append(pmid)

		#dataset: Pubmed papers
		if dataset[0] == "P":
			
			uniprot_data_list = data_list
			uniprot_class_list = class_list
			uniprot_id_list = id_list
			
			if dataset == "P3":
				fname1 = "pubmed_go_names_papers_dict_2.json"
				fname2 = "pubmed_go_names_papers_2.json"
			else:
				fname1 = "pubmed_gene_names_papers_dict.json"
				fname2 = "pubmed_gene_names_papers.json"
	
			f = open(fname1,"r")
			go_papers_dict = json.load(f)
			f.close()
			
			f = open(fname2,"r")
			pubmed_papers_dict = json.load(f)
			f.close()

			go_papers_keys = go_papers_dict.keys()
			pubmed_papers_keys = pubmed_papers_dict.keys()
			print("Pubmed papers: ", len(pubmed_papers_keys))
			for node in go_ontology:
				if node['namespace']==namespace:
					go_term = node['id']
					if dataset == "P3":
						if go_term in go_papers_keys:
							pubmed_ids = go_papers_dict[go_term]
							pubmed_ids = pubmed_ids[:5]
							for pmid in pubmed_ids:
								if pmid in pubmed_papers_keys and pmid not in uniprot_pmids:
									text = text_preprocessing(pubmed_papers_dict[pmid])
									class_list.append(go_term)
									id_list.append(pmid)
									data_list.append(text)
					else:
						proteins_go = data[data[:,1]==go_term]
						proteins_go = list(set(proteins_go[:,0]))
						for protein_id in proteins_go:
							if protein_id in go_papers_keys:
								pubmed_ids = go_papers_dict[protein_id]
								pubmed_ids = pubmed_ids[:3]
								for pmid in pubmed_ids:
									if pmid in pubmed_papers_keys and pmid not in uniprot_pmids:
										text = text_preprocessing(pubmed_papers_dict[pmid])
										class_list.append(go_term)
										id_list.append(pmid)
										data_list.append(text)
			data_list += uniprot_data_list
			class_list += uniprot_class_list
			id_list += uniprot_id_list
				

		#shuffle dataset
		(data_list, class_list, id_list) = shuffle_data(data_list, class_list, id_list)
		#divide the dataset in n folds
		total = len(data_list)
		div = int(total/folds)
		precision_kfold = []
		recall_kfold = []
		f1_kfold = []
		thresh_kfold = []
		precision_trunc_kfold = []
		recall_trunc_kfold = []
		f1_trunc_kfold = []
		thresh_trunc_kfold = []
		#file_to_write = {}
		for fold  in range(folds):
			print("Fold ", (fold+1))
			test_start = int(fold*div)
			test_end = int((fold+1)*div)
			X_test = data_list[test_start:test_end]
			class_test = class_list[test_start:test_end]
			id_test = id_list[test_start:test_end]
									
			if test_start == 0:
				train_start = div
				train_end = total
				X_train = data_list[train_start:train_end]
				class_train = class_list[train_start:train_end]
				id_train = id_list[train_start:train_end]
				
			else:
				train_start = 0
				train_end = test_start
				if test_end == total:
					X_train = data_list[train_start:train_end]
					class_train = class_list[train_start:train_end]
					id_train = id_list[train_start:train_end]
				else:
					train_start2 = test_end
					train_end2 = total
					X_train = data_list[train_start:train_end]+data_list[train_start2:train_end2]
					class_train = class_list[train_start:train_end]+class_list[train_start2:train_end2]
					id_train = id_list[train_start:train_end]+id_list[train_start2:train_end2]

			#remove duplicate papers
			(X_test, class_test, id_test) = remove_duplicate_papers(id_train, X_test, class_test, id_test, dataset)

			#vectorize features
			vectorizer = TfidfVectorizer()
			X_train = vectorizer.fit_transform(X_train)
			X_test = vectorizer.transform(X_test)

			#create binary classifiers
			print("Create classifiers")
			classifiers, classifier_keys = create_classifiers(sample_threshold, namespace)
			print("Done creating classifiers. Classifier count: ", len(classifiers))

			#consolidate test set papers with more than 1 GO label
			test_dict = {}
			for i in range(len(id_test)):
				if id_test[i] not in test_dict.keys():
					test_dict[id_test[i]] = {}
					test_dict[id_test[i]]["X"] = X_test[i]
					test_dict[id_test[i]]["y"] = []
				test_dict[id_test[i]]["y"].append(class_test[i])		
				#if id_test[i] not in file_to_write.keys():
				#	file_to_write[id_test[i]] = []

			print("Running the classifiers on the test set")
			time_start_test = time()
			prob_dict = {}
			ids = test_dict.keys()
			for i in range(len(ids)):
				test_point = test_dict[ids[i]]["X"]
				prob_dict[ids[i]] = predict_go(test_point)
			time_end_test = time()-time_start_test		
	
			print("Calculate F1/recall/precision by threshold")
			precision_list = []
			recall_list = []
			f1_list = []
			precision_trunc_list = []
			recall_trunc_list = []
			f1_trunc_list = []
			true_labels = {}
			true_labels_trunc = {}
			#true_labels_uniprot = {}
			for i in ids:
				#actual labels of each test point
				true_labels[i] = propagate_go_terms(id_test_dict[i])
				#labels of each test point excluding those dont have classifiers
				if ont == "F":
					trunc_labels = list(set(true_labels[i]) & set(classifier_keys+["GO:0008150"]))
				elif ont == "P":
					trunc_labels = list(set(true_labels[i]) & set(classifier_keys+["GO:0003674"]))
				else:
					trunc_labels = list(set(true_labels[i]) & set(classifier_keys))
				true_labels_trunc[i] = propagate_go_terms(trunc_labels)
			for r in range(1,101):
				thresh = r/100
				total_precision = 0
				total_recall = 0
				total_precision_trunc = 0
				total_recall_trunc = 0
				results_dict = {}
				for i in ids:
					all_labels = np.array(classifier_keys)
					all_prob = np.array(prob_dict[i])
					positive_labels = list(all_labels[all_prob[:]>=thresh])
					positive_filtered = [p for p in positive_labels if p not in exclude_classes]
					predicted_labels = propagate_go_terms(positive_filtered)
					if save_results=="Y" and fold == 0:
						results_dict[i] = predicted_labels
					if len(predicted_labels)>0:
						precision,recall = evaluate_prediction(true_labels[i], predicted_labels)
						total_precision+=precision
						total_recall+=recall
						precision,recall = evaluate_prediction(true_labels_trunc[i], predicted_labels)
						total_precision_trunc+=precision
						total_recall_trunc+=recall
				final_precision = total_precision/len(ids)
				final_recall = total_recall/len(ids)
				final_precision_trunc = total_precision_trunc/len(ids)
				final_recall_trunc = total_recall_trunc/len(ids)
				final_f1 = 0
				final_f1_trunc = 0
				if final_precision+final_recall>0:
					final_f1 = (2*final_precision*final_recall)/(final_precision+final_recall)
				if final_precision_trunc+final_recall_trunc>0:
					final_f1_trunc = (2*final_precision_trunc*final_recall_trunc)/(final_precision_trunc+final_recall_trunc)
				precision_list.append(final_precision)
				recall_list.append(final_recall)
				f1_list.append(final_f1)
				precision_trunc_list.append(final_precision_trunc)
				recall_trunc_list.append(final_recall_trunc)
				f1_trunc_list.append(final_f1_trunc)

			max_f1 = max(f1_list)
			max_thresh = f1_list.index(max_f1)
			max_precision = precision_list[max_thresh]
			max_recall = recall_list[max_thresh]
			f1_kfold.append(max_f1)
			thresh_kfold.append(max_thresh)
			precision_kfold.append(max_precision)
			recall_kfold.append(max_recall)

			max_f1_trunc = max(f1_trunc_list)
			max_thresh_trunc = f1_trunc_list.index(max_f1_trunc)
			max_precision_trunc = precision_trunc_list[max_thresh_trunc]
			max_recall_trunc = recall_trunc_list[max_thresh_trunc]
			f1_trunc_kfold.append(max_f1_trunc)
			thresh_trunc_kfold.append(max_thresh_trunc)
			precision_trunc_kfold.append(max_precision_trunc)
			recall_trunc_kfold.append(max_recall_trunc)

			print("Max F1 = ", max_f1)
			print("Max Precision = ", max_precision)
			print("Max Recall = ", max_recall)
			print("Maximizing Thresh = ", max_thresh)
			
			print("\nMax F1 trunc: ", max_f1_trunc)
			print("Max Precision trunc: ", max_precision_trunc)
			print("Max Recall trunc: ", max_recall_trunc)
			print("Maximizing Thresh = ", max_thresh_trunc)
		
		print("\n-----Results-----")
		print("Avg F1: ", np.mean(np.array(f1_kfold)))
		print("Avg Precision: ", np.mean(np.array(precision_kfold)))
		print("Avg Recall: ", np.mean(np.array(recall_kfold)))
		print("Avg Threshold: ", np.mean(np.array(thresh_kfold)))


		print("\nAvg F1 trunc: ", np.mean(np.array(f1_trunc_kfold)))
		print("Avg Precision trunc: ", np.mean(np.array(precision_trunc_kfold)))
		print("Avg Recall trunc: ", np.mean(np.array(recall_trunc_kfold)))
		print("Avg Threshold trunc: ", np.mean(np.array(thresh_trunc_kfold)))

		print("\n-----Settings-----")
		print("Ontology: ", ont)
		print("Dataset: ", dataset)
		print("Sample threshold: ", sample_threshold)
		print("No. of folds: ", folds)
		print("Fraction train data: ", fraction_train_data)
		print("Fraction test data: ", fraction_test_data)
		print("Total time: ", time()-time_start_all)
		
		print("\nDONE!\n")

