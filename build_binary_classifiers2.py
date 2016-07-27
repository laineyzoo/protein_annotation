from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.random as npr
import csv
import collections
import json
import string
import re
from time import time

from nltk.corpus import stopwords
from nltk import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn import linear_model
from sklearn import metrics
from sklearn import svm
import pickle

from joblib import Parallel, delayed
import multiprocessing

##CONNECT TO DB
#import sqlite3
#DB_NAME = "goa_uniprot_noiea.db"
#conn = sqlite3.connect(DB_NAME)
#c = conn.cursor()
########################

GO_ONTOLOGIES = {
    "CELLULAR_COMPONENT" : "GO:0005575",
    "BIOLOGICAL_PROCESS" : "GO:0008150",
    "MOLECULAR_FUNCTION" : "GO:0003674"
}

GO_CELL_COMPONENTS = {
    "NUCLEAR_ENVELOPE"      : "GO:0005635",
    "NUCLEOLUS"             : "GO:0005730",
    "MEMBRANE"              : "GO:0016020",
    "CELL_WALL"             : "GO:0005618",
    "CYTOSKELETON"          : "GO:0005856",
    "ENDOPLASMIC_RETICULUM" : "GO:0005783",
    "RIBOSOMES"             : "GO:0005840",
    "GOLGI_COMPLEX"         : "GO:0005794",
    "MITOCHONDRIA"          : "GO:0005739",
    "CHLOROPLAST"           : "GO:0009507",
    "VACUOLE"               : "GO:0005773",
    "PEROXISOMES"           : "GO:0005777",
    "LYSOSOMES"             : "GO:0005764",
    "CHROMOSOME"            : "GO:0005694",
    "SYNAPSE"               : "GO:0045202",
    "CELL_JUNCTION"         : "GO:0030054",
    "CELL_PROJECTION"       : "GO:0042995"

}



#extend the list of stopwords to include biological terms
bio_stopwords = ["pubmed",
                 "medline"
                 "epub",
                 "author",
                 "information",
                 "indexed"]

ext_stopwords = stopwords.words("english")
ext_stopwords.extend(bio_stopwords)


## get all GO terms that are descendants of the given GO term
def get_descendants(goterm) :
    GO_JSON = "go.json"
    f = open(GO_JSON)
    data = json.load(f)
    go_descendants = []
    go_queue = [ goterm ]
    while go_queue :
        current_term = go_queue.pop(0)
        go_descendants.append(current_term)
        for term in data :
            if current_term in term.get('is_a', []) + term.get('part_of', []) :
                if term['id'] not in go_descendants :
                    go_queue.append(term['id'])
    return go_descendants


## get only the direct descendants of the given GO term
def get_direct_descendants(go_term):
    GO_JSON = "go.json"
    f = open(GO_JSON)
    data = json.load(f)
    go_direct = list()
    for term in data:
        if go_term in term.get('is_a', []) + term.get('part_of', []):
            go_direct.append(term['id'])
    return go_direct

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
def remove_duplicate_papers(pmids_train, X_test, go_terms_test, pmids_test, ont):
	print("Remove duplicates")
	# papers in the train set should not be in the test set
	delete_indexes = list()
	for test_point in pmids_test:
		#check if this paper from the test set appears in the train set
		if test_point in pmids_train:
			indexes = list(np.where(pmids_test==test_point)[0])
			delete_indexes.extend(indexes)
	for protein in proteins:
		#get the pubmed ids of papers associated with this protein
		matching_records = data[data[:,0]==protein]
		matching_pmids = list(set(matching_records[:,2]))
		for pmid in matching_pmids:
			if pmid in pmids_train and pmid in pmids_test:
				indexes = list(np.where(pmids_test==pmid)[0])
				delete_indexes.extend(indexes)
	#delete the datapoints from the test set meeting the above two conditions
	delete_indexes = list(set(delete_indexes))
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


def softmax(x):
	return np.exp(x) / np.sum(np.exp(x), axis=0)


#BFS traversal
def traverse_ontology_bfs(parent):
	print("Traverse ontology from ", parent)
	children = get_direct_descendants(parent)
	#special case for the root's children
	if parent in GO_ONTOLOGIES.values():
		softmax_scores = ontology_softmax[parent]
		for i in range(len(children)):
			ontology_probabilities[children[i]] = softmax_scores[i]
		for i in range(len(children)):
			traverse_ontology(children[i])
	#all other nodes
	elif len(children) > 0:
		softmax_scores = ontology_softmax[parent]
		for i in range(len(children)):
			final_prob = ontology_probabilities[parent] * softmax_scores[i]
			if children[i] in ontology_probabilities.keys():
				ontology_probabilities[children[i]] = ontology_probabilities[children[i]] + final_prob
			else:
				ontology_probabilities[children[i]] = final_prob
		for i in range(len(children)):
			traverse_ontology(children[i])


#DFS traversal
def traverse_ontology_dfs(parent):
	print("Traverse ontology from ", parent)
	children = get_direct_descendants(parent)
	#special case for the root's children
	if parent in GO_ONTOLOGIES.values():
		softmax_scores = ontology_softmax[parent]
		for i in range(len(children)):
			ontology_probabilities[children[i]] = softmax_scores[i]
			traverse_ontology(children[i])
	#all other nodes
	elif len(children) > 0:
		softmax_scores = ontology_softmax[parent]
		for i in range(len(children)):
			final_prob = ontology_probabilities[parent] * softmax_scores[i]
			if children[i] in ontology_probabilities.keys():
				ontology_probabilities[children[i]] = ontology_probabilities[children[i]] + final_prob
			else:
				ontology_probabilities[children[i]] = final_prob
			traverse_ontology(children[i])


def compute_ontology_probabilities(test_point):
	nodes_seen = list()
	ontology_softmax = {}
	node_count = 0
	for node in go_ontology:
		go_id = node['id']
		namespace = node['namespace']
		if go_id not in nodes_seen and namespace == 'cellular_component':
			nodes_seen.append(go_id)
			node_count+=1
			print("Node count = ", node_count)
			children = get_direct_descendants(go_id)
			scores = list()
			for child in children:
				clf = classifiers[child]
				prob = clf.predict_proba(test_point)[0,1]
				scores.append(prob)
			softmax_scores = softmax(scores)
			ontology_softmax[go_id] = softmax_scores
	#calculate final probabilities for the whole ontology
	root = GO_ONTOLOGIES["CELLULAR_COMPONENT"]
	ontology_probabilities = {}
	traverse_ontology_bfs(root)
	#save probabilities for each test point
	outfile = open(pmids_test[i]+".txt", "ab+")
	pickle.dump(ontology_probabilities, outfile)
	outfile.close()


#######################################################################################


print("\nSTART\n")
file1 = open("protein_records.csv","r")
reader = csv.reader(file1)
data = np.array(list(reader))
data = data[data[:,4]=="C"]

file2 = open("pubmed_records.csv","r")
reader = csv.reader(file2)
data2 = np.array(list(reader))

proteins = list(set(data[:,0]))
pmids = list(set(data[:,2]))

file1.close()
file2.close()

abstracts = list()
go_terms = list()
pmids_dataset = list()

for pmid in pmids:
	matching_pub = data2[data2[:,1]==pmid]
	matching_proteins = data[data[:,2]==pmid]
	text = matching_pub[0][4]
	text = text_preprocessing(text)
	go_terms_protein = list(set(matching_proteins[:,1]))
	for term in go_terms_protein:
		abstracts.append(text)
		go_terms.append(term)
		pmids_dataset.append(pmid)

#shuffle dataset
(abstracts, go_terms, pmids_dataset) = shuffle_data(abstracts, go_terms, pmids_dataset)

#divide dataset
index = int(len(abstracts)/5)*4

X_train = abstracts[:index]
go_terms_train = go_terms[:index]
pmids_train = pmids_dataset[:index]

X_test = abstracts[index:]
go_terms_test = go_terms[index:]
pmids_test = pmids_dataset[index:]

remove_duplicate_papers(pmids_train, X_test, go_terms_test, pmids_test, "C")

#vectorize features
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5)
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

#create binary classifiers
GO_JSON = "go.json"
f = open(GO_JSON)
go_ontology = json.load(f)
f.close()

node_count = 0
nodes_seen = list()
classifier_dict = {}
for node in go_ontology:
	go_id = node['id']
	namespace = node['namespace']
	if go_id not in nodes_seen and namespace == 'cellular_component':
		node_count+=1
		nodes_seen.append(go_id)
		print("Node count: ", node_count)
		print("GO term: ", go_id)
		descendants = get_descendants(go_id)
		y_train = list()
		for term in go_terms_train:
			if term in descendants:
				y_train.append(1)
			else:
				y_train.append(0)
		classifier = MultinomialNB(alpha=.01).fit(X_train, y_train)
		classifier_dict[go_id] = classifier


output = open("classifiers_file.txt", "ab+")
pickle.dump(classifier_dict, output)
output.close()


#run binary classifiers on the test set
classifiers_file = open("classifiers_file.txt", "rb")
classifiers = pickle.load(classifiers_file)
classifiers_file.close()

ontology_softmax = {}
ontology_probabilities = {}

#parallelize this job
n_cores = 10 #multiprocessing.cpu_count() #get the no. of cores of this machine
Parallel(n_jobs=n_cores)(delayed(compute_ontology_probabilities)(test_point) for test_point in X_test)






