from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.random as npr
import csv
import collections
import json
import string
import re
import pickle
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
    text = " ".join( [ stemmer.stem(w) for w in text.split() ] )
    return text


def softmax(x):
	return np.exp(x) / np.sum(np.exp(x), axis=0)


#compute the softmax score for each child of the given parent
def compute_softmax_scores(parent, test_point):
	print("Compute softmax scores for children of ", parent)
	direct_children = get_direct_descendants(parent)
	descendant_dict = {}
	for child in direct_children:
		descendant_dict[child] = get_descendants(child)
	probabilities = list()
	for child in direct_children:
		y_train = list()
		for term in go_terms_train:
			if term in descendant_dict[child]:
				y_train.append(1)
			else:
				y_train.append(0)
		##train classifier for this node
		node_count+=1
		print("Node count: ", node_count)
		print("train MNB classifier for node ", child)
		classifier = MultinomialNB(alpha=.01).fit(X_train, y_train)
		##get the probability of success
		probabilities.append(classifier.predict_proba(test_point)[0,1])
	softmax_scores = softmax(probabilities)
	ontology_scores[parent] = softmax_scores


def traverse_ontology(parent, test_point):
	print("Traverse ontology from ", parent)
	children = get_direct_descendants(parent)
	if len(children)>0:
		compute_softmax_scores(parent, test_point)
		for child in children:
			traverse_ontology(child, test_point)
	else:
		leaf_count+=1


#########################################################################################


print("\nSTART\n")
file1 = open("protein_records.csv","r")
reader = csv.reader(file1)
data = np.array(list(reader))
data_cc = data[data[:,4]=="C"]

file2 = open("pubmed_records.csv","r")
reader = csv.reader(file2)
data2 = np.array(list(reader))

proteins_cc = list(set(data_cc[:,0]))
pmids_cc = list(set(data_cc[:,2]))

file1.close()

abstracts_cc = list()
go_terms_cc = list()
pmids_dataset = list()

for pmid in pmids_cc:
	matching_pub = data2[data2[:,1]==pmid]
	matching_proteins = data_cc[data_cc[:,2]==pmid]
	text = matching_pub[0][4]
	text = text_preprocessing(text)
	go_terms = list(set(matching_proteins[:,1]))
	for term in go_terms:
		abstracts_cc.append(text)
		go_terms_cc.append(term)
		pmids_dataset.append(pmid)

#shuffle dataset
(abstracts_cc, go_terms_cc, pmids_dataset) = shuffle_data(abstracts_cc, go_terms_cc, pmids_dataset)

#divide dataset
index = int(len(pmids_cc)/5)*4

X_train = abstracts_cc[:index]
go_terms_train = go_terms_cc[:index]
pmids_train = pmids_cc[:index]

X_test = abstracts_cc[index:]
go_terms_test = go_terms_cc[index:]
pmids_test = pmids_cc[index:]

#create vectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5)
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

#open files
GO_JSON = "go.json"
f = open(GO_JSON)
go_ontology = json.load(f)

classifiers_file = open("classifiers_output.txt", "rb")
classifiers = pickle.load(classifiers_file)

#run all classifiers for each test point
test_count = 0
for i in range(3,len(go_terms_test)):
	test_point = X_test[i]
	test_count+=1
	print("Test count: ", test_count)
	print("Test point: ", pmids_test[i])
	nodes_seen = list()
	node_count = 0
	leaf_count = 0
	ontology_scores = {}
	for node in go_ontology:
		go_id = node['id']
		namespace = node['namespace']
		if go_id not in nodes_seen and namespace == 'cellular_component':
			node_count+=1
			nodes_seen.append(go_id)
			print("Node count: ", node_count)
			print("GO id: ", go_id)
			children = get_direct_descendants(go_id)
			scores = list()
			for child in children:
				clf = classifiers[child]
				prob = clf.predict_proba(test_point)[0,1]
				scores.append(prob)
			softmax_scores = softmax(scores)
			ontology_scores[go_id] = softmax_scores
	test = open(pmids_test[i]+".txt", "ab+")
	pickle.dump(ontology_scores, test)
	test.close()
	

