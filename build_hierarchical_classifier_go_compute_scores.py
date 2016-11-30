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


## randomize the ordering of the dataset
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
def remove_duplicate_papers(pmids_train, X_test, go_terms_test, pmids_test):
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
def propagate_go_terms(go_terms):
	label_list = []
	for term in go_terms:
		label_list.extend(get_ancestors(term))
	return list(set(label_list))


def predict_go(test_point):
	prob_ontology = []
	for clf in classifiers:
		prob = clf.predict_proba(test_point)[0][1]
		prob_ontology.append(prob)
	return prob_ontology


def compute_metrics_by_protein(true_labels, true_labels_trunc):
	#Calculate F1/recall/precision by threshold for each protein
	precision_list = []
	recall_list = []
	f1_list = []
	precision_trunc_list = []
	recall_trunc_list = []
	f1_trunc_list = []
	for t in range(1,101):
		thresh = t/100
		total_precision = 0
		total_recall = 0
		total_precision_trunc = 0
		total_recall_trunc = 0
		for protein in unique_proteins:
			pmids = protein_pmid_dict[protein]
			positive_labels = []
			for i in pmids:
				all_labels = np.array(classifier_keys)
				all_prob = np.array(prob_dict[i])
				positive_labels.extend(list(all_labels[all_prob[:]>=thresh]))
			positive_labels = list(set(positive_labels))
			positive_filtered = [p for p in positive_labels if p not in exclude_classes]
			if len(positive_filtered)>0:
				predicted_labels = propagate_go_terms(positive_filtered)
				true_labels_protein = []
				true_labels_protein_trunc = []
				for i in pmids:
					true_labels_protein.extend(true_labels[i])
					true_labels_protein_trunc.extend(true_labels_trunc[i])
				true_labels_protein = list(set(true_labels_protein)
				precision, recall = evaluate_prediction(true_labels_protein, predicted_labels)
				total_precision+=precision
				total_recall+=recall
				precision, recall = evaluate_prediction(true_labels_protein_trunc, predicted_labels)
				total_precision_trunc +=precision
				total_recall_trunc +=recall
		avg_precision = total_precision/len(unique_proteins)
		avg_recall = total_recall/len(unique_proteins)
		avg_precision_trunc = total_precision_trunc/len(unique_proteins)
		avg_recall_trunc = total_recall_trunc/len(unique_proteins)
		f1 = 0
		f1_trunc = 0
		if avg_precision+avg_recall>0:
			f1 = (2*avg_precision*avg_recall/(avg_precision+avg_recall))
		if avg_precision_trunc+avg_precision_trunc>0:
			f1_trunc = (2*avg_precision_trunc*avg_recall_trunc/(avg_precision_trunc+avg_recall_trunc))
		precision_list.append(avg_precision)
		recall_list.append(avg_recall)
		f1_list.append(f1)
		precision_trunc_list.append(avg_precision_trunc)
		recall_trunc_list.append(avg_recall_trunc)
		f1_trunc_list.append(f1_trunc)
										   
	return f1_list, precision_list, recall_list, f1_trunc_list, precision_trunc_list, recall_trunc_list

										   
def compute_metrics_by_go_term(true_labels, true_labels_trunc):
	#Calculate F1/recall/precision by threshold for each GO term
	precision_list = []
	recall_list = []
	f1_list = []
	precision_trunc_list = []
	recall_trunc_list = []
	f1_trunc_list = []
	for t in range(1,101):
		thresh = t/100
		total_precision = 0
		total_recall = 0
		total_precision_trunc = 0
		total_recall_trunc = 0
		for term in unique_go_terms:
			pmids = go_pmid_dict[term]
			positive_labels = []
			for i in pmids:
				all_labels = np.array(classifier_keys)
				all_prob = np.array(prob_dict[i])
				positive_labels.extend(list(all_labels[all_prob[:]>=thresh]))
			positive_labels = list(set(positive_labels))
			positive_filtered = [p for p in positive_labels if p not in exclude_classes]
			if len(positive_filtered)>0:
				predicted_labels = propagate_go_terms(positive_filtered)
				true_labels_go = []
				true_labels_go_trunc = []
				for i in pmids:
					true_labels_go.extend(true_labels[i])
					true_labels_go_trunc.extend(true_labels_trunc[i])
					true_labels_go = list(set(true_labels_go)
				true_labels_go_trunc = list(set(true_labels_go_trunc))
				precision, recall = evaluate_prediction(true_labels_go, predicted_labels)
				total_precision+=precision
				total_recall+=recall
				precision, recall = evaluate_prediction(true_labels_go_trunc, predicted_labels)
				total_precision_trunc+=precision
				total_recall_trunc+=recall
		avg_precision = total_precision/len(unique_go_terms)
		avg_recall = total_recall/len(unique_go_terms)
		avg_precision_trunc = total_precision_trunc/len(unique_go_terms)
		avg_recall_trunc = total_recall_trunc/len(unique_go_terms)
		f1 = 0
		f1_trunc = 0
		if avg_precision+avg_recall>0:
			f1 = (2*avg_precision*avg_recall/(avg_precision+avg_recall))
		if avg_precision_trunc+avg_recall_trunc>0:
			f1_trunc = (2*avg_precision_trunc*avg_recall_trunc/(avg_precision_trunc+avg_recall_trunc))
		precision_list.append(avg_precision)
		recall_list.append(avg_recall)
		f1_list.append(f1)
		precision_trunc_list.append(avg_precision_trunc)
		recall_trunc_list.append(avg_recall_trunc)
		f1_trunc_list.append(f1_trunc)
																 
	return f1_list, precision_list, recall_list, f1_trunc_list, precision_trunc_list, recall_trunc_list
																 

										   

####################################### MAIN #############################################
if __name__ == "__main__":
	
	if len(sys.argv[1:]) < 7:
		print("This script requires at least 4 arguments: ontology, dataset, classifier, sample_threshold, save_results, compute_protein_score, compute_go_score")
		exit()
	else:
		print("=====START=====")
		print("\nSettings:")
		ont = sys.argv[1]
		namespace = "cellular_component"
		if ont == "F":
			namespace = "molecular_function"
		elif ont == "P":
			namespace = "biological_process"
		print("Ontology:", namespace)

		dataset = sys.argv[2]
		if dataset=="P1":
			print("Train set: PubMed papers w/ GO names - intersection w/ Uniprot\nTest set: Uniprot abstracts")
		elif dataset=="P2":
			print("Train set: PubMed papers w/ gene names - intersection w/ Uniprot\nTest set: Uniprot abstracts")
		elif dataset == "HP1":
			print("Train set: PubMed papers w/ GO names - intersection w/ Uniprot\nTest set: human proteins")
		elif dataset == "HP2":
			print("Train set: PubMed papers w/ gene names - intersection w/ Uniprot\nTest set: human proteins")	
                elif dataset == "YP1":
                        print("Train set: PubMed papers w/ GO names - intersection w/ Uniprot\nTest set: yeast proteins")
                elif dataset == "YP2":
                        print("Train set: PubMed papers w/ gene names - intersection w/ Uniprot\nTest set: yeast proteins")  
		else:
			print("Invalid dataset: Acceptable datasets: P1, P2, HP1, HP2, YP1, YP2. Exiting...")
			exit()			

		algo = sys.argv[3]
		if algo != "S":
			print("Classifier: Naive Bayes")
		else:
			print("Classifier: SVM")
	
		sample_threshold = int(sys.argv[4])
		print("Sample threshold: ", sample_threshold)
	
		save_results = sys.argv[5]
		print("Save results? ", save_results)

		compute_protein_score = sys.argv[6]
		print("Compute metrics by protein? ", compute_protein_score)

		compute_go_score = sys.argv[7]
		print("Compute metrics by GO term? ", compute_go_score)

		time_start_all = time()

		allowed_go_terms = go_parents.keys()
		#open the dataset files and create the dataset
		print("\nPreparing the dataset")
		data_list = list()
		class_list = list()
		id_list = list()

		f = open("protein_records.json","r")
		data = np.array(json.load(f))
		f.close()
		data = data[data[:,4]==ont]
		#remove GO:0005515 (protein binding) from dataset
		if ont == "F":
			data = data[data[:,1]!="GO:0005515"]
		unique_proteins = list(set(data[:,0]))
		uniprot_pmids = list(set(data[:,2]))

		f = open("pubmed_records.json","r")
		data2 = json.load(f)
		f.close()
		
		f = open("protein_human_records.json","r")
		human = np.array(json.load(f))
		human = human[human[:,2]!="IEA"]
		human_ids= list(set(human[:,0]))
		f = open("protein_yeast_records.json","r")
		yeast = np.array(json.load(f))
		yeast = yeast[yeast[:,2]!="IEA"]
		yeast_ids = list(set(yeast[:,0]))

		# keep list of PMIDs per protein
		protein_pmid_dict = {}
			if compute_protein_score == "Y":
				for protein in unique_proteins:
					matching_records = data[data[:,0]==protein]
					matching_pmids = list(set(matching_records[:,2]))
					protein_pmid_dict[protein] = matching_pmids
	
		#keep list of PMIDs per GO term
		go_pmid_dict = {}
		if compute_go_score == "Y":
			for term in uniprot_go_terms:
				matching_records = data[data[:,1]==term]
				matching_pmids = list(set(matching_records[:,2]))
				go_pmid_dict[term] = matching_pmids
		
		if dataset in ["P1","HP1","YP1"]:
			fname1 = "pubmed_go_names_synonym_papers_dict.json"
			fname2 = "pubmed_go_names_synonym_papers.json"
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
		print("No. of unique papers: ", len(pubmed_papers_keys))
		count=0
		for node in go_ontology:
			if node['namespace']==namespace:
				go_term = node['id']
				if dataset in ["P1", "HP1", "YP1"]:
					if go_term in go_papers_keys and go_term in allowed_go_terms:
						pubmed_ids = go_papers_dict[go_term]
						if len(pubmed_ids)>5:
							pubmed_ids = pubmed_ids[:5]
						for pmid in pubmed_ids:
							if (pmid not in uniprot_pmids and pmid in pubmed_papers_keys):
								text = pubmed_papers_dict[pmid]
								text = text_preprocessing(text)
								class_list.append(go_term)
								id_list.append(pmid)
								data_list.append(text)
				else:
					#for P2 dataset, we need to associate GO id's to protein id's first
					proteins_go = data[data[:,1]==go_term]
					proteins_go = list(set(proteins_go[:,0]))
					count+=1
					print("GO term no. ", count)
					print("GO term: ", go_term)
					print("proteins w/ this GO term: ", len(proteins_go))
					for protein_id in proteins_go:
						if protein_id in go_papers_keys:
							pubmed_ids = go_papers_dict[protein_id]
							if len(pubmed_ids)>5:
								pubmed_ids = pubmed_ids[:5]  
								for pmid in pubmed_ids:
									if (pmid not in uniprot_pmids and pmid in pubmed_papers_keys):
										text = pubmed_papers_dict[pmid]
										text = text_preprocessing(text)
										class_list.append(go_term)
										id_list.append(pmid)
										data_list.append(text)
		
		
		#shuffle dataset
		(data_list, class_list, id_list) = shuffle_data(data_list, class_list, id_list)
		#divide dataset

		fraction_train_data = 1
		if dataset in ["P1","HP1","YP1"] and ont == "C":
			index = int(len(data_list)/fraction_train_data)
		elif dataset in ["P2", "HP2","YP2"] and ont =="C":
			fraction_train_data = 2
			index = int(len(data_list)/fraction_train_data)
		elif dataset in ["P1","HP1","YP1"] and ont == "F":
			fraction_train_data = 1
			index = int(len(data_list)/fraction_train_data)
		elif dataset in ["P2", "HP2","YP2"] and ont =="F":
			fraction_train_data = 1
			index = int(len(data_list)/fraction_train_data)
		else:
			fraction_train_data = 2
			index = int(len(data_list)/fraction_train_data)
		X_train = data_list[:index]
		class_train = class_list[:index]
		id_train = id_list[:index]
		#prepare test set
		pmids_test = list(set(data[:,2]))
		X_test = []
		class_test = []
		id_test = []
		for pmid in pmids_test:
			text = text_preprocessing(data2[pmid])
			matching_proteins = data[data[:,2]==pmid]
			protein_ids = list(set(matching_proteins[:,0]))
			go_terms_protein = list(set(matching_proteins[:,1]))
			for term in go_terms_protein:  
				if (term in allowed_go_terms and dataset[0] != "H" and dataset[0]!= "Y") or ( dataset[0]=="H" and term in allowed_go_terms and len(set(protein_ids)&set(human_ids))>0 ) or ( dataset[0]=="Y" and term in allowed_go_terms and len(set(protein_ids)&set(yeast_ids))>0 ):
					X_test.append(text)
					class_test.append(term)
					id_test.append(pmid)
		#shuffle the test data aka Uniprot dataset
		(X_test, class_test, id_test) = shuffle_data(X_test, class_test, id_test)
		#take only 20% of Uniprot for testing
		fraction_test_data = 1
		if dataset[0] == "P":
			fraction_test_data = 1
			index = int(len(X_test)/fraction_test_data)
		else:
			if ont == "F":
				fraction_test_data = 1
			index = int(len(X_test)/fraction_test_data)
		X_test = X_test[:index]
		class_test = class_test[:index]
		id_test = id_test[:index]			
		
		train_len = len(X_train)
		test_len = len(X_test)	
		print("Train set: ", train_len)
		print("Fraction: ", fraction_train_data)
		print("Test set: ", test_len)
		print("Fraction: ", fraction_test_data)

		#vectorize features
		vectorizer = TfidfVectorizer()
		X_train = vectorizer.fit_transform(X_train)
		X_test = vectorizer.transform(X_test)
		
		#create binary classifiers
		print("Creating classifiers")
		time_start_classifier = time()
		classifiers = []
		classifier_keys = []
		for node in go_ontology:
			go_id = node['id']
			children = get_children(go_id)
			if node['namespace'] == namespace :
				descendants = get_descendants(go_id)
				y_train = list()
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
		print("Done creating classifiers. Classifier count: ", len(classifiers))
		time_end_classifier = time()-time_start_classifier

		#consolidate test set papers with more than 1 GO label
		id_test_dict = {}
		X_test_unique = []
		for i in range(len(id_test)):
			if id_test[i] not in id_test_dict.keys():
				id_test_dict[id_test[i]] = []
				X_test_unique.append(X_test[i])
			id_test_dict[id_test[i]].append(class_test[i])

		print("Running the classifiers on the test set")
		time_start_test = time()
		prob_dict = {}
		ids = id_test_dict.keys()
		for i in range(len(ids)):
			test_point = X_test_unique[i]
			prob_dict[ids[i]] = predict_go(test_point)
		time_end_test = time()-time_start_test

		print("Propagating true labels")
		true_labels = {}
		true_labels_trunc = {}
		for i in ids:
			#actual labels of each test point
			true_labels[i] = propagate_go_terms(id_test_dict[i])
			#labels of each test point excluding those dont have classifiers
			if ont == "F":
				trunc_labels = set(true_labels[i]) & set(classifier_keys+["GO:0008150"])
			elif ont == "P":
				trunc_labels = set(true_labels[i]) & set(classifier_keys+["GO:0003674"])
			else:
				trunc_labels = set(true_labels[i]) & set(classifier_keys)
			true_labels_trunc[i] = propagate_go_terms(trunc_labels)
		if save_results == "Y":
			results_dict = {}
			with open("results/true_labels_GO_"+ont+"_"+dataset+".json","w") as f:
				json.dump(true_labels, f)

		if compute_protein_score == "Y":
			f1_list, precision_list, recall_list, f1_trunc_list, precision_trunc_list, recall_trunc_list = compute_metrics_by_protein(true_labels, true_labels_trunc)
			max_f1 = max(f1_list)
		  	max_thresh = f1_list.index(max_f1)
		  	max_precision = precision_list[max_thresh]
		  	max_recall = recall_list[max_thresh]
		  
		  	max_f1_trunc = max(f1_trunc_list)
		  	max_thresh_trunc = f1_trunc_list.index(max_f1_trunc)
		  	max_precision_trunc = precision_trunc_list[max_thresh_trunc]
		  	max_recall_trunc = recall_trunc_list[max_thresh_trunc]
		  
		  	f1_protein_kfold.append(max_f1)
		  	thresh_protein_kfold.append(max_thresh)
		  	precision_protein_kfold.append(max_precision)
		  	recall_protein_kfold.append(max_recall)
		  
		  	f1_protein_trunc_kfold.append(max_f1_trunc)
		  	thresh_protein_trunc_kfold.append(max_thresh_trunc)
		  	precision_protein_trunc_kfold.append(max_precision_trunc)
		  	recall_protein_trunc_kfold.append(max_recall_trunc)
		  
		  	print("Max F1 = ", max_f1)
		  	print("Max Precision = ", max_precision)
		  	print("Max Recall = ", max_recall)
		  	print("Maximizing Thresh = ", max_thresh)
		  
		  	print("\nMax F1 trunc: ", max_f1_trunc)
		  	print("Max Precision trunc: ", max_precision_trunc)
		  	print("Max Recall trunc: ", max_recall_trunc)
		  	print("Maximizing Thresh = ", max_thresh_trunc)
									
		if compute_go_score == "Y":
			print("\nCalculate F1/recall/precision by threshold for GO term")
			f1_list, precision_list, recall_list, f1_trunc_list, precision_trunc_list, recall_trunc_list = compute_metrics_by_go_term(true_labels, true_labels_trunc)
		  	max_f1 = max(f1_list)
		  	max_thresh = f1_list.index(max_f1)
		  	max_precision = precision_list[max_thresh]
		  	max_recall = recall_list[max_thresh]
		  
		  	max_f1_trunc = max(f1_trunc_list)
		  	max_thresh_trunc = f1_trunc_list.index(max_f1_trunc)
		  	max_precision_trunc = precision_trunc_list[max_thresh_trunc]
		  	max_recall_trunc = recall_trunc_list[max_thresh_trunc]
		  
		  	f1_go_kfold.append(max_f1)
		  	thresh_go_kfold.append(max_thresh)
		  	precision_go_kfold.append(max_precision)
		  	recall_go_kfold.append(max_recall)
		  
		  	f1_go_trunc_kfold.append(max_f1_trunc)
		  	thresh_go_trunc_kfold.append(max_thresh_trunc)
		  	precision_go_trunc_kfold.append(max_precision_trunc)
		  	recall_go_trunc_kfold.append(max_recall_trunc)
		  
		  	print("Max F1 = ", max_f1)
		  	print("Max Precision = ", max_precision)
		  	print("Max Recall = ", max_recall)
		  	print("Maximizing Thresh = ", max_thresh)
		  
		  	print("\nMax F1 trunc: ", max_f1_trunc)
		  	print("Max Precision trunc: ", max_precision_trunc)
		  	print("Max Recall trunc: ", max_recall_trunc)
		  	print("Maximizing Thresh = ", max_thresh_trunc)
										  
										  
		print("Calculate F1/recall/precision by threshold")
		time_start_eval = time()
		precision_list = []
		recall_list = []
		f1_list = []
		precision_trunc_list = []
		recall_trunc_list = []
		f1_trunc_list = []
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
				if save_results == "Y":
					results_dict[i] = predicted_labels
				if len(predicted_labels)>0:
					precision,recall = evaluate_prediction(true_labels[i], predicted_labels)
					total_precision+=precision
					total_recall+=recall
					precision,recall = evaluate_prediction(true_labels_trunc[i], predicted_labels)
					total_precision_trunc+=precision
					total_recall_trunc+=recall
				else:
					total_precision+=0
					total_recall+=0
					total_precision_trunc+=0
					total_recall_trunc+=0
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
			if save_results=="Y":
				with open("results/predicted_labels_GO_"+ont+"_"+dataset+"_"+str(thresh)+".json","w") as f:
					json.dump(results_dict, f)
		max_f1 = max(f1_list)
		max_thresh = f1_list.index(max_f1)
		max_precision = precision_list[max_thresh]
		max_recall = recall_list[max_thresh]

		max_f1_trunc = max(f1_trunc_list)
		max_thresh_trunc = f1_trunc_list.index(max_f1_trunc)
		max_precision_trunc = precision_trunc_list[max_thresh_trunc]
		max_recall_trunc = recall_trunc_list[max_thresh_trunc]

		time_end_eval = time()-time_start_eval
		time_end_all = time()-time_start_all

		print("\n-----Results-----")
		print("Max F1: ", max_f1)
		print("Max Precision: ", max_precision)
		print("Max Recall: ", max_recall)
		print("Maximizing Threshold: ", max_thresh)
			
		print("\nMax F1 trunc: ", max_f1_trunc)
		print("Max Precision trunc: ", max_precision_trunc)
		print("Max Recall trunc: ", max_recall_trunc)
		print("Maximizing Thresh: ", max_thresh_trunc)

		print("\n-----Timings-----")
		print("Total time: ", time_end_all)

		print("\n-----Settings-----")
		print("Ontology: ", ont)
		print("Dataset: ", dataset)
		print("Script name: ", sys.argv[0])
		print("Train data: ", train_len)
                print("Fraction: ", fraction_train_data)
		print("Test data: ", test_len)
		print("Fraction: ", fraction_test_data)
		print("\nDONE!\n")

		#save output to file
		f = open("log_GO_"+ont+"_"+dataset+"_"+str(test_len)+".txt","w")
		print("\n-----Results-----", file=f)
		print("Max F1: ", max_f1, file=f)
		print("Max Precision: ", max_precision, file=f)
		print("Max Recall: ", max_recall, file=f)
		print("Max Threshold: ", max_thresh, file=f)

		print("\nMax F1 trunc: ", max_f1_trunc, file=f)
		print("Max Precision trunc: ", max_precision_trunc, file=f)
		print("Max Recall trunc: ", max_recall_trunc, file=f)
		print("Maximizing Thresh: ", max_thresh_trunc, file=f)

		print("\n-----Timings-----", file=f)
		print("Creating classifiers: ", time_end_classifier, file=f)
		print("Evaluating test set: ", time_end_test, file=f)
		print("Computing metrics per threshold: ", time_end_eval, file=f)
		print("Total time: ", time_end_all, file=f)

		print("\n-----Settings-----", file=f)
		print("Ontology: ", ont, file=f)
		print("Dataset: ", dataset, file=f)
		print("Classifier: ", algo, file=f)
		print("Sample threshold: ", sample_threshold, file=f)
		print("No. of classifiers: ", len(classifiers), file=f)
		print("Train data: ", train_len,file=f)
		print("Fraction: ", fraction_train_data,file=f)
		print("Test data: ", test_len,file=f)
		print("Fraction: ", fraction_test_data,file=f)
		print("Script name: ", sys.argv[0], file=f)
		print("\nDONE!\n", file=f)
