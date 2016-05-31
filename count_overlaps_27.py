from __future__ import division

import numpy as np
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
            go_direct.append(term)

    return go_direct



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

#display the classifier performance
def print_metrics(actual, predicted):

    class_names = sorted(list(GO_CELL_COMPONENTS.keys()))
    class_names.append("OTHER")
    performance = metrics.classification_report(actual, predicted, target_names=class_names)
    print >> outfile, "\nPerformance:"
    print >> outfile, performance
    print(performance)

    accuracy = metrics.accuracy_score(actual, predicted)
    print >> outfile, "Accuracy: "+str(accuracy*100)
    print("Accuracy: ", accuracy*100)

    conf_matrix = metrics.confusion_matrix(actual, predicted)
    print >> outfile, "\nConfusion Matrix:"
    print >> outfile, conf_matrix
    print(conf_matrix)    

    f1 = metrics.precision_recall_fscore_support(actual, predicted)
    f1 = f1[2].mean(axis=0)
    
    return (accuracy, f1)


# remove duplicate papers/proteins appearing in both train and test
def remove_duplicate_papers(X_train, y_train, pmids_train, X_test, y_test, pmids_test):

    print "Remove duplicates"
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



#training and testing phase with 4 types of classifiers
def train_and_test(X_train, y_train, X_test, y_test):
    
    print "Vectorizing features"
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5)
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)
    
    print >> outfile, "\nn_samples: %d, n_features: %d" % X_train.shape
    
    #print("PCA decomposition")
    #pca = decomposition.PCA()
    #X_train_pca = pca.fit_transform(X_train.toarray())
    #X_test_pca = pca.transform(X_test.toarray())
    
    #print("\nPCA n_samples: %d, n_features: %d" % X_train_pca.shape)
    
    print >> outfile, "Train Set Counts:"
    counts = collections.Counter(y_train)
    print >> outfile, "Total: ", len(y_train)
    print >> outfile, counts
    
    print >> outfile, "Test Set Counts:"
    counts = collections.Counter(y_test)
    print >> outfile, "Total: ", len(y_test)
    print >> outfile, counts
    
    print "Training and Testing"
	
    print >> outfile, "\n=====MULTINOMIAL NAIVE BAYES====="
    print("\n=====MULTINOMIAL NAIVE BAYES=====")
    t0 = time()
    classifier = MultinomialNB(alpha=.01)
    classifier.fit(X_train, y_train)
    predicted = classifier.predict(X_test)
		
    acc_mnb, f1_mnb = print_metrics(y_test, predicted)
    print >> outfile, "\nTime to train (min.): ", (time()-t0)/60
    
    
    print >> outfile, "\n========SVM========"
    print("\n========SVM========")
    t0 = time()
    classifier = svm.LinearSVC(multi_class='ovr')
    classifier.fit(X_train, y_train)
    predicted = classifier.predict(X_test)
	
    acc_bnb, f1_bnb = print_metrics(y_test, predicted)
    print >> outfile, "\nTime to train (min.): ", (time()-t0)/60


    print("\n========Neural Network========")
    t0 = time()
    classifier = MLPClassifier(algorithm='l-bfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    classifier.fit(X_train, y_train)
    predicted = classifier.predict(X_test)

    acc_nn, f1_nn = print_metrics(y_test, predicted)
    print >> outfile, "\nTime to train (min.): ", (time()-t0)/60
	

   
    print("\n=====RANDOM FOREST=====")
    t0 = time()
    classifier = RandomForestClassifier(n_estimators=400, n_jobs=10)
    classifier.fit(X_train, y_train)
    predicted = classifier.predict(X_test)
    
    acc_forest, f1_forest = print_metrics(y_test, predicted)
    print >> outfile, "\nTime to train (min.): ", (time()-t0)/60
    
    
    print >> outfile, "\n=====ADABOOST====="
    print("\n=====ADABOOST=====")
    t0 = time()
    base_classifier = svm.SVC(decision_function_shape='ovr')
    base_classifier.fit(X_train, y_train)
    
    ada_real = AdaBoostClassifier(base_estimator=base_classifier,learning_rate=1,n_estimators=400,algorithm="SAMME")
    ada_real.fit(X_train, y_train)
    predicted = ada_real.predict(X_test)
    
    acc_ada, f1_ada = print_metrics(y_test, predicted)
    print >> outfile, "\nTime to train (min.): ", (time()-t0)/60
	

    return (acc_forest, f1_forest, acc_ada, f1_ada, acc_mnb, f1_mnb, acc_bnb, f1_bnb)



#K-fold validation (train and testing are called from here)
def k_fold_validation(K, labels, abstracts, pmids_dataset):
    
    labels_folds = np.array_split(labels, K)
    abstracts_folds = np.array_split(abstracts, K)
    pmids_dataset_folds = np.array_split(pmids_dataset, K)
    
    n_classifiers = 4
    accuracy_mat = np.empty([K, n_classifiers], dtype=float)
    f1_mat = np.empty([K, n_classifiers], dtype=float)
    
    for fold in range(K):
        
        print >> outfile, "\n============ Fold " + str(fold) + " ==============\n"
        print("\n============ Fold " + str(fold) + " ==============\n")
	
        k_fold_labels = np.delete(labels_folds,fold,axis=0)
        k_fold_labels = np.hstack(k_fold_labels)
        k_fold_labels = list(k_fold_labels)
        
        k_fold_abstracts= np.delete(abstracts_folds,fold,axis=0)
        k_fold_abstracts = np.hstack(k_fold_abstracts)
        k_fold_abstracts = list(k_fold_abstracts)
        
        pmids_array = np.delete(pmids_dataset_folds,fold,axis=0)
        k_fold_pmids = np.hstack(pmids_array)
        k_fold_pmids = list(k_fold_pmids)
        
        X_train = k_fold_abstracts
        y_train = k_fold_labels
        pmids_train = k_fold_pmids
        
        X_test = list(abstracts_folds[fold])
        y_test = list(labels_folds[fold])
        pmids_test = list(pmids_dataset_folds[fold])
        
        #remove duplicates from the test set
        (X_test, y_test) = remove_duplicate_papers(k_fold_abstracts, k_fold_labels, k_fold_pmids, X_test, y_test, pmids_test)
        
        #start training and testing
        (acc1, f1_1,  acc2, f1_2, acc3, f1_3, acc4, f1_4) = train_and_test(k_fold_abstracts, k_fold_labels, X_test, y_test)
        
        accuracy_mat[fold][0] = acc1*100
        accuracy_mat[fold][1] = acc2*100
        accuracy_mat[fold][2] = acc3*100
        accuracy_mat[fold][3] = acc4*100
        f1_mat[fold][0] = f1_1*100
        f1_mat[fold][1] = f1_2*100
        f1_mat[fold][2] = f1_3*100
        f1_mat[fold][3] = f1_4*100

    avg_acc = accuracy_mat.mean(axis=0)
    avg_f1 = f1_mat.mean(axis=0)

    print >> outfile, "\n=============== SUMMARY REPORT ==============="
    
    print >> outfile, "\nAverage Accuracy:\n"
    print >> outfile, "Random Forest: ", avg_acc[0]
    print >> outfile, "AdaBoost: ", avg_acc[1]
    print >> outfile, "Multinomial NB: ", avg_acc[2]
    print >> outfile, "SVM: ", avg_acc[3]
    
    print >> outfile, "\nAverage F1-score:\n"
    print >> outfile, "Random Forest: ", avg_f1[0]
    print >> outfile, "AdaBoost: ", avg_f1[1]
    print >> outfile, "Multinomial NB: ", avg_f1[2]
    print >> outfile, "SVM: ", avg_f1[3]
    
    print >> outfile, "\nTotal time duration: ", (time() - time_start)/60


################################################################


print "Getting GO and PubMed records from file"

time_start = time()
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

outfile = open("outfile.txt","w+")

print >> outfile, "Start time", time_start
## get the cell components descendants ##
# structure of the dict:
# descendant_dict["NUCLEUS"] = ['GO:0000', 'GO:0001', etc]

descendant_dict = {}
cell_components = sorted(list(GO_CELL_COMPONENTS.keys()))
for cell_comp in cell_components:
    descendant_dict[cell_comp] = get_descendants(GO_CELL_COMPONENTS[cell_comp])

#class labels:
# 0-15: all keys in GO_CELL_COMPONENTS (15) + none of the above (1)

print "Getting abstracts and assigning labels"
print >> outfile, "Getting abstracts and assigning labels"

labels = list()
abstracts = list()
pmids_dataset = list() #every datapoint will have an associated pubmed id in this list

for pmid in pmids:

    matching_proteins = data[data[:,2]==pmid]
    go_terms = list(set(matching_proteins[:,1]))
    matching_pub = data2[data2[:,1]==pmid]
    text = matching_pub[0][4]
    text = text_preprocessing(text)
    
    for term in go_terms:
        
        belongs_to_component = False
        
        for i in range(len(cell_components)):
            if term in descendant_dict[cell_components[i]]:
                labels.append(i)
                abstracts.append(text)
                pmids_dataset.append(pmid)
                belongs_to_component = True
    
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
proportion.append((float)(len(intersect)/len(ids[0]))*100)

#overlaps for component #1
concat = np.concatenate((ids[0], ids[2], ids[3], ids[4], ids[5], ids[6], ids[7], ids[8], ids[9], ids[10], ids[11], ids[12], ids[13], ids[14], ids[15], ids[16]), axis=0)
intersect = np.intersect1d(ids[1], concat)
intersection.append(len(intersect))
proportion.append((float)(len(intersect)/len(ids[1]))*100)

#overlaps for component #2
concat = np.concatenate((ids[0], ids[1], ids[3], ids[4], ids[5], ids[6], ids[7], ids[8], ids[9], ids[10], ids[11], ids[12], ids[13], ids[14], ids[15], ids[16]), axis=0)
intersect = np.intersect1d(ids[2], concat)
intersection.append(len(intersect))
proportion.append((float)(len(intersect)/len(ids[2]))*100)

#overlaps for component #3
concat = np.concatenate((ids[0], ids[1], ids[2], ids[4], ids[5], ids[6], ids[7], ids[8], ids[9], ids[10], ids[11], ids[12], ids[13], ids[14], ids[15], ids[16]), axis=0)
intersect = np.intersect1d(ids[3], concat)
intersection.append(len(intersect))
proportion.append((float)(len(intersect)/len(ids[3]))*100)

#overlaps for component #4
concat = np.concatenate((ids[0], ids[1], ids[2], ids[3], ids[5], ids[6], ids[7], ids[8], ids[9], ids[10], ids[11], ids[12], ids[13], ids[14], ids[15], ids[16]), axis=0)
intersect = np.intersect1d(ids[4], concat)
intersection.append(len(intersect))
proportion.append((float)(len(intersect)/len(ids[4]))*100)

#overlaps for component #5
concat = np.concatenate((ids[0], ids[1], ids[2], ids[3], ids[4], ids[6], ids[7], ids[8], ids[9], ids[10], ids[11], ids[12], ids[13], ids[14], ids[15], ids[16]), axis=0)
intersect = np.intersect1d(ids[5], concat)
intersection.append(len(intersect))
proportion.append((float)(len(intersect)/len(ids[5]))*100)

#overlaps for component #6
concat = np.concatenate((ids[0], ids[1], ids[2], ids[3], ids[4], ids[5], ids[7], ids[8], ids[9], ids[10], ids[11], ids[12], ids[13], ids[14], ids[15], ids[16]), axis=0)
intersect = np.intersect1d(ids[6], concat)
intersection.append(len(intersect))
proportion.append((float)(len(intersect)/len(ids[6]))*100)

#overlaps for component #7
concat = np.concatenate((ids[0], ids[1], ids[2], ids[3], ids[4], ids[5], ids[6], ids[8], ids[9], ids[10], ids[11], ids[12], ids[13], ids[14], ids[15], ids[16]), axis=0)
intersect = np.intersect1d(ids[7], concat)
intersection.append(len(intersect))
proportion.append((float)(len(intersect)/len(ids[7]))*100)

#overlaps for component #8
concat = np.concatenate((ids[0], ids[2], ids[3], ids[4], ids[5], ids[6], ids[7], ids[1], ids[9], ids[10], ids[11], ids[12], ids[13], ids[14], ids[15], ids[16]), axis=0)
intersect = np.intersect1d(ids[8], concat)
intersection.append(len(intersect))
proportion.append((float)(len(intersect)/len(ids[8]))*100)

#overlaps for component #9
concat = np.concatenate((ids[0], ids[2], ids[3], ids[4], ids[5], ids[6], ids[7], ids[8], ids[1], ids[10], ids[11], ids[12], ids[13], ids[14], ids[15], ids[16]), axis=0)
intersect = np.intersect1d(ids[9], concat)
intersection.append(len(intersect))
proportion.append((float)(len(intersect)/len(ids[9]))*100)

#overlaps for component #10
concat = np.concatenate((ids[0], ids[2], ids[3], ids[4], ids[5], ids[6], ids[7], ids[8], ids[9], ids[1], ids[11], ids[12], ids[13], ids[14], ids[15], ids[16]), axis=0)
intersect = np.intersect1d(ids[10], concat)
intersection.append(len(intersect))
proportion.append((float)(len(intersect)/len(ids[10]))*100)

#overlaps for component #11
concat = np.concatenate((ids[0], ids[2], ids[3], ids[4], ids[5], ids[6], ids[7], ids[8], ids[9], ids[10], ids[1], ids[12], ids[13], ids[14], ids[15], ids[16]), axis=0)
intersect = np.intersect1d(ids[11], concat)
intersection.append(len(intersect))
proportion.append((float)(len(intersect)/len(ids[11]))*100)

#overlaps for component #12
concat = np.concatenate((ids[0], ids[2], ids[3], ids[4], ids[5], ids[6], ids[7], ids[8], ids[9], ids[10], ids[11], ids[1], ids[13], ids[14], ids[15], ids[16]), axis=0)
intersect = np.intersect1d(ids[12], concat)
intersection.append(len(intersect))
proportion.append((float)(len(intersect)/len(ids[12]))*100)

#overlaps for component #13
concat = np.concatenate((ids[0], ids[2], ids[3], ids[4], ids[5], ids[6], ids[7], ids[8], ids[9], ids[10], ids[11], ids[12], ids[1], ids[14], ids[15], ids[16]), axis=0)
intersect = np.intersect1d(ids[13], concat)
intersection.append(len(intersect))
proportion.append((float)(len(intersect)/len(ids[13]))*100)

#overlaps for component #14
concat = np.concatenate((ids[0], ids[2], ids[3], ids[4], ids[5], ids[6], ids[7], ids[8], ids[9], ids[10], ids[11], ids[12], ids[13], ids[1], ids[15], ids[16]), axis=0)
intersect = np.intersect1d(ids[14], concat)
intersection.append(len(intersect))
proportion.append((float)(len(intersect)/len(ids[14]))*100)

#overlaps for component #15
concat = np.concatenate((ids[0], ids[2], ids[3], ids[4], ids[5], ids[6], ids[7], ids[8], ids[9], ids[10], ids[11], ids[12], ids[13], ids[14], ids[1], ids[16]), axis=0)
intersect = np.intersect1d(ids[15], concat)
intersection.append(len(intersect))
proportion.append((float)(len(intersect)/len(ids[15]))*100)

#overlaps for component #16
concat = np.concatenate((ids[0], ids[2], ids[3], ids[4], ids[5], ids[6], ids[7], ids[8], ids[9], ids[10], ids[11], ids[12], ids[13], ids[14], ids[15], ids[16]), axis=0)
intersect = np.intersect1d(ids[16], concat)
intersection.append(len(intersect))
proportion.append((float)(len(intersect)/len(ids[16]))*100)

sorted_index = sorted(range(len(proportion)), key=lambda k: proportion[k])

for i in sorted_index:
	print("COMPONENT: ", cell_components[i])
	print("COUNT: ", len(ids[i]))
	print("OVERLAP: ", intersection[i])
	print("PERCENT: ", proportion[i])

print("DONE!\n")








