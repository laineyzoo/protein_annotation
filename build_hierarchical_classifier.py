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


from nltk.corpus import stopwords
from nltk import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm


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


#extend the list of stopwords to include biological terms
bio_stopwords = ["pubmed",
                 "medline"
                 "epub",
                 "author",
                 "information",
                 "indexed"]

ext_stopwords = stopwords.words("english")
ext_stopwords.extend(bio_stopwords)


def get_descendants(goterm) :
	go_descendants = []
	go_queue = [ goterm ]
	while go_queue :
		current_term = go_queue.pop(0)
		go_descendants.append(current_term)
		for term in go_ontology :
			term_id = term['id']
			if current_term in go_parents[term_id]:
				if term_id not in go_descendants :
					go_queue.append(term_id)
	return list(set(go_descendants))



## get only the direct descendants of the given GO term
def get_direct_descendants(go_term):
    go_direct = list()
    for term in go_ontology:
        if go_term in term.get('is_a', []) + term.get('part_of', []):
            go_direct.append(term['id'])
    return go_direct


def get_children(go_term):
	return go_children[go_term]

def get_parents(go_term):
	return go_parents[go_term]

## given GO id, return the entry for the GO term
def get_node(node_id):
	return (item for item in go_ontology if item["id"]==node_id).next()

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
	for protein in unique_proteins:
		#get the pubmed ids of papers associated with this protein
		matching_records = data[data[:,0]==protein]
		matching_pmids = list(set(matching_records[:,2]))
		for pmid in matching_pmids:
			if pmid in pmids_train and pmid in pmids_test:
				indexes = list(np.where(pmids_test_array==pmid)[0])
				delete_indexes.extend(indexes)
	#delete the datapoints from the test set meeting the above two conditions
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
    no_stopwords = [word for word in text.split() if word.lower() not in ext_stopwords]
    text = " ".join(no_stopwords)
    #stem the words
    stemmer = PorterStemmer()
    text = " ".join([stemmer.stem(w) for w in text.split()])
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


# propagate the GO annotation of each  test point upwards in the ontology until it reaches the root
def propagate_go_terms(go_terms):
	label_list = []
	for term in go_terms:
		labels = list()
		labels.append(term)
		q = collections.deque()
		q.append(term)
		#traverse ontology upwards from node to root
		while len(q)>0:
			node = q.popleft()
			parents = get_parents(node)
			labels.extend(parents)
			q.extend(parents)
		#remove duplicates in the label set
		labels = list(set(labels))
		#add this label set to our list
		label_list.extend(labels)
	label_list = list(set(label_list))
	return label_list

#predict one or more go annotations for this instance
def predict_go(test_point):
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



####################################### MAIN #############################################
if __name__ == "__main__":
	
	print("\nSTART\n")
	if len(argv[1:]) < 3:
		print("This script requires at least 3 arguments")
		exit()
	else:
		ont = argv[1]
		print("Ontology: ", ont)
		namespace = "cellular_component"
		if ont == "F":
			namespace = "molecular_function"
		elif ont == "P":
			namespace = "biological_process"
		
		dataset = argv[2]
		print("Dataset: ", dataset)

		algo = argv[3]
		if algo == "B":
			print("Classifier: Naive Bayes")
		elif algo == "S":
			print("Classifier: SVM")

		kmer = 5
		print("k-mer size: ", kmer)

		time_start_all = time()
		#open the dataset files and create the dataset
		print("Preparing the dataset")
		data_list = list()
		class_list = list()
		id_list = list()

		f = open("protein_records.csv","r")
		reader = csv.reader(f)
		data = np.array(list(reader))
		data = protein_records[data[:,4]==ont]
		f.close()
		unique_proteins = list(set(data[:,0]))

		#dataset: UniProt abstracts
		if dataset == "U":
			f = open("pubmed_records.csv","r")
			reader = csv.reader(f)
			data2 = np.array(list(reader))
			f.close()

			pmids = list(set(data[:,2]))
			for pmid in pmids:
				matching_pub = data2[data2[:,1]==pmid]
				matching_proteins = data[data[:,2]==pmid]
				text = matching_pub[0][4]
				text = text_preprocessing(text)
				go_terms_protein = list(set(matching_proteins[:,1]))
				for term in go_terms_protein:
					data_list.append(text)
					class_list.append(term)
					id_list.append(pmid)

		#dataset: Protein sequences
		elif dataset == "S":
			f = open("protein_sequences.json","r")
			seq_dict = json.load(f)
			f.close()

			seq_dict_keys = seq_dict.keys()
			for prot_id in unique_proteins:
				if prot_id in seq_dict_keys:
					matching_proteins = data[data[:,0]==prot_id]
					go_terms_protein = list(set(matching_proteins[:,1]))
					seq = seq_dict[prot_id]
					seq = create_kmers(seq, k)
					for term in go_terms_protein:
						data_list.append(seq)
						class_list.append(term)
						id_list.append(prot_id)

		#dataset: Pubmed w/ GO names
		elif dataset == "P1" or dataset=="P2":
			if dataset == "P1":
				fname1 = "pubmed_go_names_papers_dict.json"
				fname2 = "pubmed_go_names_papers.json"
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
			for node in go_ontology:
				if node['namespace']==namespace:
					go_term = node['id']
					if go_term in go_papers_keys:
						pubmed_ids = go_papers_dict[go_term]
						if len(pubmed_ids)>5:
							pubmed_ids = pubmed_ids[:5]
						for pmid in pubmed_ids:
							if pmid in pubmed_papers_keys:
								text = pubmed_papers_dict[pmid]
								text = text_preprocessing(text)
								class_list.append(go_term)
								id_list.append(pmid)
								data_list.append(text)

		#shuffle dataset
		(data_list, class_list, id_list) = shuffle_data(data_list, class_list, id_list)
		#divide dataset
		index = int(len(abstracts)/5)*4

		X_train = data_list[:index]
		class_train = class_list[:index]
		id_train = id_list[:index]

		X_test = data_list[index:]
		class_test = class_list[index:]
		id_test = id_list[index:]

		if dataset == "U":
			(X_test, class_test, id_test) = remove_duplicate_papers(id_train, X_test, class_test, id_test)

		#vectorize features
		vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5)
		X_train = vectorizer.fit_transform(X_train)
		X_test = vectorizer.transform(X_test)

		#create binary classifiers
		print("Creating classifiers")
		time_start_classifier = time()
		sample_threshold = 1
		classifiers = {}
		positive_count = {}
		for node in go_ontology:
			go_id = node['id']
			children = get_children(go_id)
			if (node['namespace'] == namespace and namespace != 'biological_process') or (namespace=='biological_process' and len(children)>=2) :
				descendants = get_descendants(go_id)
				y_train = list()
				for term in go_terms_train:
					if term in descendants:
						y_train.append(1)
					else:
						y_train.append(0)
				if y_train.count(1)>=sample_threshold and y_train.count(1) != len(y_train):
					if algo == "B":
						clf = MultinomialNB(alpha=.01).fit(X_train, y_train)
					else:
						clf = svm.SVC(probability=True)
						clf.fit(X_train, y_train)
					classifiers[go_id] = clf
					positive_count[go_id] = y_train.count(1)
				elif y_train.count(1) == len(y_train):
					clf = MultinomialNB(alpha=.01).fit(X_train, y_train)
					classifiers[go_id] = clf
					positive_count[go_id] = len(y_train)
		print("Done creating classifiers. Classifier count: ", len(classifiers))
		time_end_classifier = time()-time_start_classifier

		#consolidate test set papers with more than 1 GO label
		id_test_dict = {}
		X_test_unique = []
		for i in range(len(id_test)):
			if id_test[i] not in id_test_dict.keys():
				id_test_dict[id_test[i]] = []
				X_test_unique.append(X_test[i])
			id_test_dict[id_test[i]].append(go_terms_test[i])

		print("Running the classifiers on the test set")
		time_start_test = time()
		prob_dict = {}
		ids = id_test_dict.keys()
		for i in range(len(ids)):
			test_point = X_test_unique[i]
			prob_dict[ids[i]] = predict_go(test_point)
		time_end_test = time()-time_start_test

		print("Calculate F1/recall/precision by threshold")
		time_start_eval = time()
		precision_list = list()
		recall_list = list()
		f1_list = list()
		true_labels = {}
		for id in ids:
			true_labels[id] = propagate_go_terms(pmids_test_dict[id])
		for thresh in range(0,101):
			thresh = float(thresh)/100
			print("threshold = ", thresh)
			total_precision = 0
			total_recall = 0
			for pmid in pmids:
				positive_labels = list()
				prob_ontology = prob_dict[pmid]
				for key in classifiers.keys():
					if prob_ontology[key] >= thresh:
						positive_labels.append(key)
				predicted_labels = propagate_go_terms(positive_labels)
				precision,recall = evaluate_prediction(true_labels[pmid], predicted_labels)
				total_precision+=precision
				total_recall+=recall
			final_precision = total_precision/len(pmids)
			final_recall = total_recall/len(pmids)
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

		print("\n-----Options-----")
		print("Ontology: ", ont)
		print("Dataset: ", dataset)
		print("Classifier: ", algo)
		print("Sample threshold: ", sample_threshold)
		print("No. of classifiers: ", len(classifiers))
		print("\nDONE!\n")
