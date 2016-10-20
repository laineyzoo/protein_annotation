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
	#no need for unique protein checking for Pubmed dataset
	if dataset == "U":
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
	
	if len(sys.argv[1:]) < 4:
		print("This script requires at least 3 arguments: ontology, dataset, classifier type, sample threshold")
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
			print("Dataset: Uniprot")
		elif dataset == "P5":
			print("Dataset: PubMed papers w/ GO names AND Uniprot")
		elif dataset=="P6":
			print("Dataset: PubMed papers w/ gene names AND Uniprot")
		else:
			print("Invalid dataset. Valid datasets are U, P5 and P6. Exiting...")
			exit()

		algo = sys.argv[3]
		if algo != "S":
			print("Classifier: Naive Bayes")
		else:
			print("Classifier: SVM")
	
		sample_threshold = int(sys.argv[4])
		print("Sample threshold: ", sample_threshold)		

		#dubious annotations - don't create classifiers for these
		exclude_classes = ["GO:0005515","GO:0003674","GO:0008150","GO:0005575","GO:0005829","GO:0005737","GO:0005576","GO:0005886"]		
		
		#open the dataset files and create the dataset
		print("\nPreparing the dataset")
		data_list = list()
		class_list = list()
		id_list = list()

		f = open("protein_records.json","r")
		data = np.array(json.load(f))
		f.close()
		data = data[data[:,4]==ont]
		if ont == "F":
			data = data[data[:,1]!="GO:0005515"]
		unique_proteins = list(set(data[:,0]))

		f = open("pubmed_records.json","r")
		data2 = np.array(json.load(f))
		f.close()
		uniprot_pmids = list(set(data[:,2]))

		#dataset: UniProt abstracts
		for pmid in uniprot_pmids:
			matching_pub = data2[data2[:,1]==pmid]
			matching_proteins = data[data[:,2]==pmid]
			text = matching_pub[0][4]
			text = text_preprocessing(text)
			go_terms_protein = list(set(matching_proteins[:,1]))
			for term in go_terms_protein:
				data_list.append(text)
				class_list.append(term)
				id_list.append(pmid)

		#dataset: Uniprot papers
		if dataset != "U":
			
			uniprot_data_list = data_list
			uniprot_class_list = class_list
			uniprot_id_list = id_list
			
			if dataset == "P5":
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
			print("GO names w/ papers: ", len(go_papers_keys))
			print("Pubmed papers: ", len(pubmed_papers_keys))
			for node in go_ontology:
				if node['namespace']==namespace:
					go_term = node['id']
					if dataset == "P5":
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
					elif dataset == "P6":
						proteins_go = data[data[:,1]==go_term]
						proteins_go = list(set(data[:,0]))
						for protein_id in proteins_go:
							if protein_id in go_papers_keys:
								pubmed_ids = go_papers_dict[protein_id]
								if len(pubmed_ids)>5:
									pubmed_ids = pubmed_ids[:5]
								for pmid in pubmed_ids:
									if pmid in pubmed_papers_keys:
										text = pubmed_papers_dict[pmid]
										text = text_preprocessing(text)
										class_list.append(go_term)
										id_list.append(pmid)
										data_list.append(text)
			data_list += uniprot_data_list
			class_list += uniprot_class_list
			id_list += uniprot_id_list
				

		#shuffle dataset
		(data_list, class_list, id_list) = shuffle_data(data_list, class_list, id_list)
		#divide the dataset in 5 folds
		folds = 5
		total = len(data_list)
		div = int(total/folds)
		precision_kfold = list()
		recall_kfold = list()
		f1_kfold = list()
		thresh_kfold = list()
		for f  in range(folds):
			print("Fold ", (f+1))
			test_start = int(f*div)
			test_end = int((f+1)*div)
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
			
			#Pubmed data: For CC and MF, use 50% of the data. For BP, use 1/3
			if dataset[0]=="P":
				if ont in ["C","F"]:
					index = int(len(X_train)/2)
                        	else:
					index = int(len(X_train)/3)
				X_train = X_train[:index] 
                        	class_train = class_train[:index]
                        	id_train = id_train[:index]
			#remove duplicate papers
			(X_test, class_test, id_test) = remove_duplicate_papers(id_train, X_test, class_test, id_test, dataset)
			
			print("Train set: ", len(X_train))
			print("Test set: ", len(X_test))
			#vectorize features
			vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5)
			X_train = vectorizer.fit_transform(X_train)
			X_test = vectorizer.transform(X_test)

			#create binary classifiers
			print("Creating classifiers")
			classifiers = {}
			positive_count = {}
			for node in go_ontology:
				go_id = node['id']
				children = get_children(go_id)
				if go_id not in exclude_classes and ( (node['namespace'] == namespace and namespace != 'biological_process') or (namespace=='biological_process' and len(children)>=2) ):
					descendants = get_descendants(go_id)
					y_train = list()
					for term in class_train:
						if term in descendants:
							y_train.append(1)
						else:
							y_train.append(0)
					pos_count = y_train.count(1)
					if pos_count>=sample_threshold:
						if algo != "S" or pos_count == len(y_train):
							clf = MultinomialNB(alpha=.01).fit(X_train, y_train)
						else:
							clf = svm.SVC(probability=True).fit(X_train, y_train)
						classifiers[go_id] = clf
						positive_count[go_id] = pos_count

			print("Done creating classifiers. Classifier count: ", len(classifiers))

			#consolidate test set papers with more than 1 GO label
			id_test_dict = {}
			X_test_unique = []
		
			for i in range(len(id_test)):
				if dataset == "U":
					if id_test[i] not in id_test_dict.keys():
						id_test_dict[id_test[i]] = []
						X_test_unique.append(X_test[i])
					id_test_dict[id_test[i]].append(class_test[i])
				else:
					#for Pubmed datasets, only test Uniprot papers
					if id_test[i] in uniprot_pmids:
						if id_test[i] not in id_test_dict.keys():
							id_test_dict[id_test[i]] = []
							X_test_unique.append(X_test[i])
						id_test_dict[id_test[i]].append(class_test[i])

			print("Running the classifiers on the test set")
			prob_dict = {}
			ids = id_test_dict.keys()
			for i in range(len(ids)):
				test_point = X_test_unique[i]
				prob_dict[ids[i]] = predict_go(test_point)

			print("Calculate F1/recall/precision by threshold")
			precision_list = list()
			recall_list = list()
			f1_list = list()
			true_labels = {}
			for id in ids:
				true_labels[id] = propagate_go_terms(id_test_dict[id])
			for thresh in range(0,101):
				thresh = float(thresh)/100
				total_precision = 0
				total_recall = 0
				total_labelled = len(ids)
				for id in ids:
					positive_labels = list()
					prob_ontology = prob_dict[id]
					for key in classifiers.keys():
						if prob_ontology[key] >= thresh:
							positive_labels.append(key)
					positive_filtered = [p for p in positive_labels if p not in exclude_classes]
					if len(positive_filtered)>0:
						predicted_labels = propagate_go_terms(positive_labels)
						precision,recall = evaluate_prediction(true_labels[id], predicted_labels)
						total_precision+=precision
						total_recall+=recall
					else:
						total_labelled-=1
				final_precision = total_precision/total_labelled
				final_recall = total_recall/total_labelled
				final_f1 = (2*final_precision*final_recall)/(final_precision+final_recall)
				precision_list.append(final_precision)
				recall_list.append(final_recall)
				f1_list.append(final_f1)
			max_f1 = max(f1_list)
			max_thresh = f1_list.index(max_f1)
			max_precision = precision_list[max_thresh]
			max_recall = recall_list[max_thresh]
			f1_kfold.append(max_f1)
			thresh_kfold.append(max_thresh)
			precision_kfold.append(max_precision)
			recall_kfold.append(max_recall)
			
		
		print("\n-----Results-----")
		print("Avg F1: ", np.mean(np.array(f1_kfold)))
		print("Avg Precision: ", np.mean(np.array(precision_kfold)))
		print("Avg Recall: ", np.mean(np.array(recall_kfold)))
		print("Avg Threshold: ", np.mean(np.array(thresh_kfold)))


		print("\n-----Settings-----")
		print("Ontology: ", ont)
		print("Dataset: ", dataset)
		print("Classifier: ", algo)
		print("Sample threshold: ", sample_threshold)
		print("No. of folds: ", folds)

		print("Total time: ", time()-time_start_all)
		
		print("\nDONE!\n")


		logfile = "log_go_kfold_"+ont+"_"+dataset+"_"+algo+"_"+str(sample_threshold)+".txt"
		f = open(logfile,"w")
		print("\n-----Results-----")
		print("Avg F1: ", np.mean(np.array(f1_kfold)), file=f)
		print("Avg Precision: ", np.mean(np.array(precision_kfold)), file=f)
		print("Avg Recall: ", np.mean(np.array(recall_kfold)), file=f)
		print("Avg Threshold: ", np.mean(np.array(thresh_kfold)), file=f)
		for j in range(len(f1_kfold)):
			print("\nFold ", j, file=f)
			print("Max F1: ", f1_kfold[j])
			print("Max Precision: ", precision_kfold[j])
			print("Max Recall: ", recall_kfold[j])
			print("Best Threshold: ", thresh_kfold[j])
		
		
		print("\n-----Settings-----")
		print("Ontology: ", ont, file=f)
		print("Dataset: ", dataset, file=f)
		print("Classifier: ", algo, file=f)
		print("Sample threshold: ", sample_threshold, file=f)
		print("No. of folds: ", folds, file=f)
		
		print("Total time: ", time()-time_start_all, file=f)
