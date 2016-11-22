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


#predict one or more go annotations for this instance
def predict_go(test_point):
	prob_ontology = []
	for clf in classifiers:
		prob = clf.predict_proba(test_point)[0]
		classes = clf.classes_
		if len(classes)==2:
			positive_prob = prob[classes[:]==1][0]
		else:
			if classes[0]==0:
				positive_prob = 0.0
			else:
				positive_prob = 1.0
		prob_ontology.append(positive_prob)
	return prob_ontology



####################################### MAIN #############################################
if __name__ == "__main__":
	
	if len(sys.argv[1:]) < 6:
		print("This script requires at least 6 arguments: ontology, dataset, classifier type, sample threshold, no. of folds, save_results")
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
			print("Train set: Uniprot abstracts \nTest set: Uniprot abstracts")
		elif dataset == "H1":
			print("Train set: Uniprot")
			print("Test set: human proteins")
		elif dataset == "H2":
			print("Train set: human proteins")
			print("Test set: human proteins")
		elif dataset == "Y1":
                        print("Train set: Uniprot")
                        print("Test set: yeast proteins")	
		elif dataset == "Y2":
                        print("Train set: yeast proteins")
                        print("Test set: yeast proteins")		
		elif dataset == "P3":
			print("Dataset: PubMed papers w/ GO names AND Uniprot")
		elif dataset=="P4":
			print("Dataset: PubMed papers w/ gene names AND Uniprot")
		else:
			print("Invalid dataset. Valid datasets are U, H1, H2, Y1, Y2, P3 and P4. Exiting...")
			exit()

		algo = sys.argv[3]
		if algo != "S":
			print("Classifier: Naive Bayes")
		else:
			print("Classifier: SVM")
	
		sample_threshold = int(sys.argv[4])
		print("Sample threshold: ", sample_threshold)		
		
		folds = int(sys.argv[5])
		print("Folds: ", folds)
		
		save_results = sys.argv[6]
		print("Save results? ", save_results)

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

if dataset[0] == "H":
	f = open("protein_human_records.json","r")
	human = np.array(json.load(f))
	human = human[human[:,2]!="IEA"]
	human_ids= list(set(human[:,0]))
	f = open("protein_yeast_records.json","r")
	yeast = np.array(json.load(f))
	yeast = yeast[yeast[:,2]!="IEA"]
	yeast_ids = list(set(yeast[:,0]))

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
		if (dataset not in ["H2","Y2"] and term in allowed_go_terms) or (dataset=="H2" and len(set(protein_ids)&set(human_ids))>0 and term in allowed_go_terms) or (dataset=="Y2" and len(set(protein_ids)&set(yeast_ids))>0 and term in allowed_go_terms):
			data_list.append(text)
			class_list.append(term)
			id_list.append(pmid)

		#dataset: Pubmed papers
		if dataset[0] == "P":
			
			uniprot_data_list = data_list
			uniprot_class_list = class_list
			uniprot_id_list = id_list
			
			if dataset == "P3":
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
			
			#For CC and MF P3, use 50% of the data. For BP and P4 use 1/5
			if dataset[0] == "P":
				if dataset[0]=="P3" and ont in ["C","F"]:
					index = int(len(X_train)/2)
				else:
					index = int(len(X_train)/5)
					if ont == "P":
						index = int(len(X_train)/10)
                                        	index_test = int(len(X_test)/10)
                                        	X_test = X_test[:index_test]
                                        	class_test = class_test[:index_test]
                                        	id_test = id_test[:index_test]
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
			classifiers = []
			classifier_keys = []
			for node in go_ontology:
				go_id = node['id']
				children = get_children(go_id)
				if node['namespace'] == namespace:
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

			#consolidate test set papers with more than 1 GO label
			id_test_dict = {}
			X_test_unique = []
			for i in range(len(id_test)):
				if dataset in ["U","H2","Y2"]:
					if id_test[i] not in id_test_dict.keys():
						id_test_dict[id_test[i]] = []
						X_test_unique.append(X_test[i])
					id_test_dict[id_test[i]].append(class_test[i])
				elif dataset[0] == "P":
					#for Pubmed datasets, only test Uniprot papers
					if id_test[i] in uniprot_pmids:
						if id_test[i] not in id_test_dict.keys():
							id_test_dict[id_test[i]] = []
							X_test_unique.append(X_test[i])
						id_test_dict[id_test[i]].append(class_test[i])
				elif dataset in ["H1","Y1"]:
					#for human/yeast data, only test papers w/ human/yeast-related GO annotations
					matching_proteins = data[data[:,2]==id_test[i]]
                        		protein_ids = list(set(matching_proteins[:,0]))
					if (dataset == "H1" and len(set(protein_ids) & set(human_ids))>0) or (dataset == "Y1" and len(set(protein_ids) & set(yeast_ids))>0):
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
				else:
					trunc_labels = list(set(true_labels[i]) & set(classifier_keys))
				true_labels_trunc[i] = trunc_labels
			if save_results == "Y":
				with open("results/true_labels_kfold_GO_"+ont+"_"+dataset+"_"+algo+"_full.json","w") as f:
					json.dump(true_labels, f)
				with open("results/true_labels_kfold_GO_"+ont+"_"+dataset+"_"+algo+"_trunc.json","w") as f:
					json.dump(trunc_labels,f)
			for thresh in range(0,101):
				thresh = float(thresh)/100
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
					else:
						total_precision+=0
						total_recall+=0
						total_precision_trunc+=0
						total_recall_trunc+=0
				final_precision = total_precision/len(ids)
				final_recall = total_recall/len(ids)
				final_precision_trunc = total_precision_trunc/len(ids)
				final_recall_trunc = total_recall_trunc/len(ids)
				if final_precision+final_recall>0:
					final_f1 = (2*final_precision*final_recall)/(final_precision+final_recall)
				else:
					final_f1 = 0
				if final_precision_trunc+final_recall_trunc>0:
					final_f1_trunc = (2*final_precision_trunc*final_recall_trunc)/(final_precision_trunc+final_recall_trunc)
				else:
					final_f1_trunc = 0
				precision_list.append(final_precision)
				recall_list.append(final_recall)
				f1_list.append(final_f1)
				precision_trunc_list.append(final_precision_trunc)
				recall_trunc_list.append(final_recall_trunc)
				f1_trunc_list.append(final_f1_trunc)
				if save_results=="Y" and fold == 0 and thresh > 0.0:
					with open("results/predicted_labels_kfold_GO_"+ont+"_"+dataset+"_"+algo+"_"+str(thresh)+".json","w") as f:
						json.dump(results_dict, f)
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
		print("Classifier: ", algo)
		print("Sample threshold: ", sample_threshold)
		print("No. of folds: ", folds)

		print("Total time: ", time()-time_start_all)
		
		print("\nDONE!\n")


		logfile = "log_GO_kfold_"+ont+"_"+dataset+"_"+algo+"_"+str(sample_threshold)+"_"+str(folds)+"_trunc.txt"
		f = open(logfile,"w")
		print("\n-----Results-----", file=f)
		print("Avg F1: ", np.mean(np.array(f1_kfold)), file=f)
		print("Avg Precision: ", np.mean(np.array(precision_kfold)), file=f)
		print("Avg Recall: ", np.mean(np.array(recall_kfold)), file=f)
		print("Avg Threshold: ", np.mean(np.array(thresh_kfold)), file=f)

		print("\nAvg F1 trunc: ", np.mean(np.array(f1_trunc_kfold)), file=f)
		print("Avg Precision trunc: ", np.mean(np.array(precision_trunc_kfold)), file=f)
		print("Avg Recall trunc: ", np.mean(np.array(recall_trunc_kfold)), file=f)
		print("Avg Threshold trunc: ", np.mean(np.array(thresh_trunc_kfold)), file=f)
		
		for j in range(len(f1_kfold)):
			print("\nFold ", j, file=f)
			print("Max F1: ", f1_kfold[j],file=f)
			print("Max Precision: ", precision_kfold[j],file=f)
			print("Max Recall: ", recall_kfold[j],file=f)
			print("Best Threshold: ", thresh_kfold[j],file=f)

			print("Max F1: ", f1_trunc_kfold[j],file=f)
			print("Max Precision: ", precision_trunc_kfold[j],file=f)
			print("Max Recall: ", recall_trunc_kfold[j],file=f)
			print("Best Threshold: ", thresh_trunc_kfold[j],file=f)
		
		print("\n-----Settings-----", file=f)
		print("Ontology: ", ont, file=f)
		print("Dataset: ", dataset, file=f)
		print("Classifier: ", algo, file=f)
		print("Sample threshold: ", sample_threshold, file=f)
		print("No. of folds: ", folds, file=f)
		
		print("Total time: ", time()-time_start_all, file=f)



