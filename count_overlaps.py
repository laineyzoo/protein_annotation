
import numpy as np
import csv
import collections
import json
import string
import re
import sqlite3
from time import time

from nltk.corpus import stopwords
from nltk import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sknn.mlp import Classifier, Layer
from sklearn import linear_model
from sklearn import metrics
from sklearn import svm

##CONNECT TO DB
DB_NAME = "goa_uniprot_noiea.db"
conn = sqlite3.connect(DB_NAME)
c = conn.cursor()

#dictionary of important GO terms
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
            go_direct.append(term['name'])
    return go_direct


## get definition of GO term
def define_go_term(go_term):
    GO_JSON = "go.json"
    f = open(GO_JSON)
    data = json.load(f)
    for term in data:
        if go_term == term.get('id'):
            return term.get('name')


## randomize the ordering of the dataset ##
def shuffle_data(labels, abstracts, pmids_dataset):
    
    print("Shuffle dataset")
    
    labels_shuffle = []
    abstracts_shuffle = []
    pmids_shuffle = []
    
    index_shuffle = np.arange(len(labels))
    np.random.shuffle(index_shuffle)
    
    for i in index_shuffle:
        labels_shuffle.append(labels[i])
        abstracts_shuffle.append(abstracts[i])
        pmids_shuffle.append(pmids_dataset[i])
    
    return (labels_shuffle, abstracts_shuffle, pmids_shuffle)



## preprocess text before using it for training/testing
## preprocessing includes the ff:
## 1. change case to lowercase
## 2. remove stopwords
## 3. stem the words

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


# remove duplicate papers/proteins appearing in both train and test
def remove_duplicate_papers(X_train, y_train, pmids_train, X_test, y_test, pmids_test):

    print("Remove duplicates")
    # papers in the train set should not be in the test set
    delete_indexes = list()

    for test_point in pmids_test:
    
        #check if this paper from the test set appears in the train set
        if test_point in pmids_train:
            indexes = list(np.where(pmids_test==test_point)[0])
            delete_indexes.extend(indexes)


    # papers associated with the same protein should not be in both train and test sets
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
        del y_test[loc]

    return X_test, y_test




################################################################


print("Getting GO and PubMed records from file")

time_start = time()
file1 = open("protein_records.csv","r")
reader = csv.reader(file1)
data = np.array(list(reader))
data = data[data[:,4]=="C"]
proteins = list(set(data[:,0]))
pmids = list(set(data[:,2]))
file1.close()

## get the cell components descendants ##
# structure of the dict:
# descendant_dict["NUCLEUS"] = ['GO:0000', 'GO:0001', etc]

descendant_dict = {}
cell_components = sorted(list(GO_CELL_COMPONENTS.keys()))
for cell_comp in cell_components:
	descendant_dict[cell_comp] = get_descendants(GO_CELL_COMPONENTS[cell_comp])

#class labels:
# 0-15: all keys in GO_CELL_COMPONENTS (15) + none of the above (1)

print("Getting abstracts and assigning labels")

labels = list()
abstracts = list()
pmids_dataset = list() #every datapoint will have an associated pubmed id in this list
other_components = list()

for pmid in pmids:

    matching_proteins = data[data[:,2]==pmid]
    go_terms = list(set(matching_proteins[:,1]))
    c.execute("SELECT abstract FROM publications WHERE pubmed_id=" + pmid + " LIMIT 1")
    conn.commit()
    handle = c.fetchall()
    text = handle[0][0]
    text = text_preprocessing(text)
    
    for term in go_terms:
        
        for i in range(len(cell_components)):
            if term in descendant_dict[cell_components[i]]:
				#print("Component: ", cell_components[i])
                labels.append(i)
                abstracts.append(text)
                pmids_dataset.append(pmid)


#for each component, the number of overlaps in other components
ids = list()
intersection = list()
proportion = list()
labels = np.array(labels)
pmids_dataset = np.array(pmids_dataset)

for i in range(len(cell_components)):
	ids_component = pmids_dataset[labels[:]==i]
	ids.append(ids_component)


#overlaps for component #0
concat = np.concatenate((ids[1], ids[2], ids[3], ids[4], ids[5], ids[6], ids[7], ids[8], ids[9], ids[10], ids[11], ids[12], ids[13], ids[14], ids[15], ids[16]), axis=0)
intersect = np.intersect1d(ids[0], concat)
intersection.append(len(intersect))
proportion.append((len(intersect)/len(ids[0]))*100)

#overlaps for component #1
concat = np.concatenate((ids[0], ids[2], ids[3], ids[4], ids[5], ids[6], ids[7], ids[8], ids[9], ids[10], ids[11], ids[12], ids[13], ids[14], ids[15], ids[16]), axis=0)
intersect = np.intersect1d(ids[1], concat)
intersection.append(len(intersect))
proportion.append((len(intersect)/len(ids[1]))*100)

#overlaps for component #2
concat = np.concatenate((ids[0], ids[1], ids[3], ids[4], ids[5], ids[6], ids[7], ids[8], ids[9], ids[10], ids[11], ids[12], ids[13], ids[14], ids[15], ids[16]), axis=0)
intersect = np.intersect1d(ids[2], concat)
intersection.append(len(intersect))
proportion.append((len(intersect)/len(ids[2]))*100)

#overlaps for component #3
concat = np.concatenate((ids[0], ids[1], ids[2], ids[4], ids[5], ids[6], ids[7], ids[8], ids[9], ids[10], ids[11], ids[12], ids[13], ids[14], ids[15], ids[16]), axis=0)
intersect = np.intersect1d(ids[3], concat)
intersection.append(len(intersect))
proportion.append((len(intersect)/len(ids[3]))*100)

#overlaps for component #4
concat = np.concatenate((ids[0], ids[1], ids[2], ids[3], ids[5], ids[6], ids[7], ids[8], ids[9], ids[10], ids[11], ids[12], ids[13], ids[14], ids[15], ids[16]), axis=0)
intersect = np.intersect1d(ids[4], concat)
intersection.append(len(intersect))
proportion.append((len(intersect)/len(ids[4]))*100)

#overlaps for component #5
concat = np.concatenate((ids[0], ids[1], ids[2], ids[3], ids[4], ids[6], ids[7], ids[8], ids[9], ids[10], ids[11], ids[12], ids[13], ids[14], ids[15], ids[16]), axis=0)
intersect = np.intersect1d(ids[5], concat)
intersection.append(len(intersect))
proportion.append((len(intersect)/len(ids[5]))*100)

#overlaps for component #6
concat = np.concatenate((ids[0], ids[1], ids[2], ids[3], ids[4], ids[5], ids[7], ids[8], ids[9], ids[10], ids[11], ids[12], ids[13], ids[14], ids[15], ids[16]), axis=0)
intersect = np.intersect1d(ids[6], concat)
intersection.append(len(intersect))
proportion.append((len(intersect)/len(ids[6]))*100)

#overlaps for component #7
concat = np.concatenate((ids[0], ids[1], ids[2], ids[3], ids[4], ids[5], ids[6], ids[8], ids[9], ids[10], ids[11], ids[12], ids[13], ids[14], ids[15], ids[16]), axis=0)
intersect = np.intersect1d(ids[7], concat)
intersection.append(len(intersect))
proportion.append((len(intersect)/len(ids[7]))*100)

#overlaps for component #8
concat = np.concatenate((ids[0], ids[2], ids[3], ids[4], ids[5], ids[6], ids[7], ids[1], ids[9], ids[10], ids[11], ids[12], ids[13], ids[14], ids[15], ids[16]), axis=0)
intersect = np.intersect1d(ids[8], concat)
intersection.append(len(intersect))
proportion.append((len(intersect)/len(ids[8]))*100)

#overlaps for component #9
concat = np.concatenate((ids[0], ids[2], ids[3], ids[4], ids[5], ids[6], ids[7], ids[8], ids[1], ids[10], ids[11], ids[12], ids[13], ids[14], ids[15], ids[16]), axis=0)
intersect = np.intersect1d(ids[9], concat)
intersection.append(len(intersect))
proportion.append((len(intersect)/len(ids[9]))*100)

#overlaps for component #10
concat = np.concatenate((ids[0], ids[2], ids[3], ids[4], ids[5], ids[6], ids[7], ids[8], ids[9], ids[1], ids[11], ids[12], ids[13], ids[14], ids[15], ids[16]), axis=0)
intersect = np.intersect1d(ids[10], concat)
intersection.append(len(intersect))
proportion.append((len(intersect)/len(ids[10]))*100)

#overlaps for component #11
concat = np.concatenate((ids[0], ids[2], ids[3], ids[4], ids[5], ids[6], ids[7], ids[8], ids[9], ids[10], ids[1], ids[12], ids[13], ids[14], ids[15], ids[16]), axis=0)
intersect = np.intersect1d(ids[11], concat)
intersection.append(len(intersect))
proportion.append((len(intersect)/len(ids[11]))*100)

#overlaps for component #12
concat = np.concatenate((ids[0], ids[2], ids[3], ids[4], ids[5], ids[6], ids[7], ids[8], ids[9], ids[10], ids[11], ids[1], ids[13], ids[14], ids[15], ids[16]), axis=0)
intersect = np.intersect1d(ids[12], concat)
intersection.append(len(intersect))
proportion.append((len(intersect)/len(ids[12]))*100)

#overlaps for component #13
concat = np.concatenate((ids[0], ids[2], ids[3], ids[4], ids[5], ids[6], ids[7], ids[8], ids[9], ids[10], ids[11], ids[12], ids[1], ids[14], ids[15], ids[16]), axis=0)
intersect = np.intersect1d(ids[13], concat)
intersection.append(len(intersect))
proportion.append((len(intersect)/len(ids[13]))*100)

#overlaps for component #14
concat = np.concatenate((ids[0], ids[2], ids[3], ids[4], ids[5], ids[6], ids[7], ids[8], ids[9], ids[10], ids[11], ids[12], ids[13], ids[1], ids[15], ids[16]), axis=0)
intersect = np.intersect1d(ids[14], concat)
intersection.append(len(intersect))
proportion.append((len(intersect)/len(ids[14]))*100)

#overlaps for component #15
concat = np.concatenate((ids[0], ids[2], ids[3], ids[4], ids[5], ids[6], ids[7], ids[8], ids[9], ids[10], ids[11], ids[12], ids[13], ids[14], ids[1], ids[16]), axis=0)
intersect = np.intersect1d(ids[15], concat)
intersection.append(len(intersect))
proportion.append((len(intersect)/len(ids[15]))*100)

#overlaps for component #16
concat = np.concatenate((ids[0], ids[2], ids[3], ids[4], ids[5], ids[6], ids[7], ids[8], ids[9], ids[10], ids[11], ids[12], ids[13], ids[14], ids[15], ids[16]), axis=0)
intersect = np.intersect1d(ids[16], concat)
intersection.append(len(intersect))
proportion.append((len(intersect)/len(ids[16]))*100)

sorted_index = sorted(range(len(proportion)), key=lambda k: proportion[k])

for i in sorted_index:
	print("COMPONENT: ", cell_components[i])
	print("COUNT: ", len(ids[i]))
	print("OVERLAP: ", intersection[i])
	print("PERCENT: ", proportion[i])

print("DONE!\n")




