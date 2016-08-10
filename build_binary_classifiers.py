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
from sets import Set

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
	#GO_JSON = "go.json"
	#f = open(GO_JSON)
	#data = json.load(f)
    go_descendants = []
    go_queue = [ goterm ]
    while go_queue :
        current_term = go_queue.pop(0)
        go_descendants.append(current_term)
        for term in go_ontology :
            if current_term in term.get('is_a', []) + term.get('part_of', []) :
                if term['id'] not in go_descendants :
                    go_queue.append(term['id'])
    return go_descendants


def get_descendants_2(goterm) :
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
	return go_descendants



## get only the direct descendants of the given GO term
def get_direct_descendants(go_term):
	#GO_JSON = "go.json"
	#f = open(GO_JSON)
	#data = json.load(f)
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
def remove_duplicate_papers(pmids_train, X_test, go_terms_test, pmids_test, ont):
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
	for protein in proteins:
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
	precision = len(inter)/len(pred_labels)
	recall = len(inter)/len(true_labels)
	return precision,recall


# propagate the GO annotation of each  test point upwards in the ontology until it reaches the root
def propagate_go_terms(go_terms):
	print("Propagate GO terms")
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
	return label_list

#predict one or more go annotations for this instance
def predict_go(test_point, thresh):
	print("Predict GO")
	positive_labels = list()
	for node in go_ontology:
		if node['namespace'] == 'molecular_function':
			node_id = node['id']
			clf = classifiers[node_id]
			prob = clf.predict_proba(test_point)[0,1]
			if prob >= thresh:
				positive_labels.append(node_id)
	predicted_labels = propagate_go_terms(positive_labels)
	return predicted_labels



def create_binary_classifiers():
	global classifiers
	node_count = 0
	nodes_seen = list()
	for node in go_ontology:
		go_id = node['id']
		namespace = node['namespace']
		if go_id not in nodes_seen and namespace == 'molecular_function':
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
			clf = MultinomialNB(alpha=.01).fit(X_train, y_train)
			classifiers[go_id] = clf
	print("Done creating classifiers!")

#######################################################################################


print("\nSTART\n")
f = open("protein_records.csv","r")
reader = csv.reader(f)
data = np.array(list(reader))
data = data[data[:,4]=="F"]
f.close()

f = open("pubmed_records.csv","r")
reader = csv.reader(f)
data2 = np.array(list(reader))
f.close()

proteins = list(set(data[:,0]))
pmids = list(set(data[:,2]))


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

(X_test, go_terms_test, pmids_test) = remove_duplicate_papers(pmids_train, X_test, go_terms_test, pmids_test, "F")

#vectorize features
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5)
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

#create binary classifiers
node_count = 0
classifiers = {}
for node in go_ontology:
	go_id = node['id']
	namespace = node['namespace']
	if namespace == 'molecular_function':
		node_count+=1
		print("Node count: ", node_count)
		print("GO term: ", go_id)
		descendants = get_descendants(go_id)
		y_train = list()
		for term in go_terms_train:
			if term in descendants:
				y_train.append(1)
			else:
				y_train.append(0)
		clf = MultinomialNB(alpha=.01).fit(X_train, y_train)
		classifiers[go_id] = clf
print("Done creating classifiers!")

pmids_test_dict = {}
X_test_unique = []
for i in range(len(pmids_test)):
	if pmids_test[i] not in pmids_test_dict.keys():
		pmids_test_dict[pmids_test[i]] = []
		X_test_unique.append(X_test[i])
	pmids_test_dict[pmids_test[i]].append(go_terms_test[i])

#run binary classifiers on the test set
print("Running binary classifiers")
for x in range(55,100,5):
	thresh = x/100
	print("thresh = ", thresh)
	total_precision = 0
	total_recall = 0
	pmids = pmids_test_dict.keys()
	for i in range(len(pmids)):
		test_point = X_test_unique[i]
		true_labels = propagate_go_terms(pmids_test_dict[pmids[i]])
		predicted_labels = predict_go(X_test,thresh)
		precision,recall = evaluate_prediction(true_labels, predicted_labels)
		total_precision+=precision
		total_recall+=recall
	final_precision = total_precision/len(pmids_test)
	final_recall = total_recall/len(pmids_test)
	final_f1 = 2*((final_precision*final_recall)/(final_precision+final_recall))
	print("\nThreshold: ", thresh)
	print("Precsion: ", final_precision)
	print("Recall: ", final_recall)
	print("F1: ", final_f1)




#############################################
# Unused functions							#
#############################################



def softmax(x):
	return np.exp(x) / np.sum(np.exp(x), axis=0)


#BFS traversal - iterative, uses a set data struct to visit a node exactly once
def traverse_ontology_bfs_set(root,pmid):
	ontology_probabilities = {}
	nodes_seen = Set()
	q = collections.deque()
	q.append(root)
	while len(q)>0:
		parent = q.popleft()
		nodes_seen.add(parent)
		children = get_direct_descendants(parent)
		softmax_scores = ontology_softmax[parent]
		for i in range(len(children)):
			parents = list(set(get_parents(children[i])))
			parents_seen = nodes_seen.intersection(parents)
			if len(parents_seen) == len(parents):
				ontology_probabilities[children[i]] = 0
				for p in parents:
					if p == root:
						ontology_probabilities[children[i]] += softmax_scores[i]
					else:
						ontology_probabilities[children[i]] += (ontology_probabilities[p] * softmax_scores[i])
				if ontology_probabilities[children[i]] > 0.2:
					print("node: ", children[i])
					print("parents: ", parents)
					print("softmax: ", softmax_scores[i])
					for p in parents:
						if p != root:
							print("parent prob: ", ontology_probabilities[p])
					print("prob: ", ontology_probabilities[children[i]])
					raise AssertionError()
				q.append(children[i])
	#save probabilities for each test point
	outfile = open(pmid+"_set.txt", "ab+")
	pickle.dump(ontology_probabilities, outfile)
	outfile.close()



#BFS traversal - iterative (non-recursive)
def traverse_ontology_bfs_iterative(root,pmid):
	ontology_probabilities = {}
	q = collections.deque()
	q.append(root)
	while len(q)>0:
		parent = q.popleft()
		print("Parent: ", parent)
		children = get_direct_descendants(parent)
		softmax_scores = ontology_softmax[parent]
		if parent==root:
			for i in range(len(children)):
				ontology_probabilities[children[i]] = {}
				ontology_probabilities[children[i]][parent] = softmax_scores[i]
				q.append(children[i])
		else:
			for i in range(len(children)):
				parent_prob = sum(ontology_probabilities[parent].values())
				new_prob = parent_prob * softmax_scores[i]
				print("New prob = ", new_prob)
				if new_prob > 1:
					print("child id: ", children[i])
					print("parent id: ", parent)
					print("Prob = ", new_prob)
					raise AssertionError()
				if children[i] not in ontology_probabilities.keys():
					ontology_probabilities[children[i]] = {}
				ontology_probabilities[children[i]][parent] = new_prob
				q.append(children[i])
	#save probabilities for each test point
	outfile = open(pmid+".txt", "ab+")
	pickle.dump(ontology_probabilities, outfile)
	outfile.close()



def compute_ontology_probabilities(test_point, pmid):
	global ontology_softmax
	ontology_softmax = {}
	nodes_seen = list()
	print("Computing softmax for each node")
	for node in go_ontology:
		go_id = node['id']
		namespace = node['namespace']
		if go_id not in nodes_seen and namespace == 'molecular_function':
			nodes_seen.append(go_id)
			children = get_direct_descendants(go_id)
			scores = list()
			for child in children:
				clf = classifiers[child]
				prob = clf.predict_proba(test_point)[0,1]
				scores.append(prob)
			softmax_scores = softmax(scores)
			ontology_softmax[go_id] = softmax_scores
	#calculate final probabilities for the whole ontology
	print("Computing final probabilities")
	root = GO_ONTOLOGIES["CELLULAR_COMPONENT"]
	traverse_ontology_bfs_set(root, pmid)


def compute_raw_probabilities(test_point, pmid):
	ontology_raw_prob = {}
	nodes_seen = list()
	print("Computing raw probabilities for each node")
	for node in go_ontology:
		node_id = node['id']
		namespace = node['namespace']
		if node_id not in nodes_seen and namespace == 'cellular_component':
			nodes_seen.append(node_id)
			clf = classifiers[node_id]
			prob = clf.predict_proba(test_point)[0,1]
			print("Positive prob = ", prob)
			ontology_raw_prob[node_id] = prob
	#store raw probabilities on file
	print("Done!")
	with open(pmid+"_raw_prob_p.json", "w") as f:
		json.dump(ontology_raw_prob, f)



#return the no. of edges from the root to the given node
def get_distance_from_root(root):
	ontology_distance = {}
	q = collections.deque()
	q.append(root)
	nodes_seen = list()
	while len(q)>0:
		parent = q.popleft()
		children = get_direct_descendants(parent)
		for child in children:
			if child not in nodes_seen:
				if parent == root:
					ontology_distance[child] = 1
				else:
					ontology_distance[child] = ontology_distance[parent]+1
				nodes_seen.append(child)
				q.append(child)
	#store the distances on file
	f = open("go_ontology_distances_p.txt", "ab+")
	pickle.dump(ontology_distance, f)
	f.close()


def plot_distance_to_probability():
	f = open("pmids.txt")
	pmids = list(f)
	f = open("go_terms.txt")
	go_terms = list(f)
	f = open("distances.txt","rb")
	distances = pickle.load(f)
	
	for i in range(len(pmids)):
		dist = list()
		prob = list()
		true_prob = 0
		true_dist = 0
		true_label = go_terms[i].strip()
		with open(pmids[i].strip()+"_raw_prob.json", "r") as f:
			data = json.load(f)
		for k in data.keys():
			prob.append(data[k])
			dist.append(distances[k])
			if k == true_label:
				true_dist = distances[k]
				true_prob = data[k]
		plt.plot(dist, prob, 'o', color='b')
		plt.plot(true_dist, true_prob, 'o', color='r')
		plt.show()

def convert_pickle_to_json():
	f = open("pmids.txt","rb")
	pmids = pickle.load(f)
	
	for pmid in pmids:
		f = open(pmid.strip()+"_raw_prob.txt","rb")
		raw_prob = pickle.load(f)
		with open(pmid+"_raw_prob.json", "w") as f:
			json.dump(raw_prob, f)



#sum the probabilities of the leaves, should be approx. 1
def sum_leaves(probabilities):
	leaves_sum = 0
	count = 0
	for node in go_ontology:
		if node['namespace'] == 'cellular_component':
			node_id = node['id']
			children = get_direct_descendants(node_id)
			if len(children)==0:
				count+=1
				print("leaf count: ", count)
				leaves_sum+= sum(probabilities[node_id].values())
				print("sum = ", leaves_sum)
	#if leaves_sum > 1:
#raise AssertionError()
return leaves_sum


def sum_leaves_set(probabilities):
	leaves_sum = 0
	count = 0
	for node in go_ontology:
		if node['namespace'] == 'cellular_component':
			node_id = node['id']
			children = get_direct_descendants(node_id)
			if len(children)==0:
				count+=1
				print("leaf count: ", count)
				leaves_sum+= probabilities[node_id]
				print("sum = ", leaves_sum)
	return leaves_sum


def store_children():
	children_dict = {}
	for node in go_ontology:
		node_id = node['id']
		children = get_direct_descendants(node_id)
		children_dict[node_id] = children
	with open("go_children.json","w") as f:
		json.dump(children_dict, f)


def store_parents():
	parent_dict = {}
	for node in go_ontology:
		node_id = node['id']
		if 'part_of' in node.keys():
			parents = list(set(node['is_a']+node['part_of']))
		elif 'is_a' in node.keys():
			parents = node['is_a']
		else:
			parents = []
		print("Node: ", node_id)
		print("Parents: ", parents)
		parent_dict[node_id] = parents
	with open("go_parents.json","w") as f:
		json.dump(parent_dict, f)



