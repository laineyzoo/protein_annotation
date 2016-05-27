
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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import KFold
from sklearn import linear_model
from sklearn import metrics
from sklearn import svm



#dictionary of important GO terms
GO_ONTOLOGIES = {
    "CELLULAR_COMPONENT" : "GO:0005575",
    "BIOLOGICAL_PROCESS" : "GO:0008150",
    "MOLECULAR_FUNCTION" : "GO:0003674"
}

PRIMARY_CELL_COMPONENTS = {
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
    "LYSOSOMES"             : "GO:0005764"
}

PRIMARY_CELL_COMPONENTS_ID = {
    "NUCLEUS"               : 0,
    "CYTOPLASM"             : 1,
    "MEMBRANE"              : 2,
    "CELL_WALL"             : 3,
    "CYTOSKELETON"          : 4
}


CYTOPLASM_CELL_COMPONENTS = {
    "ENDOPLASMIC_RETICULUM" : "GO:0005783",
	"RIBOSOMES"             : "GO:0005840",
	"GOLGI_COMPLEX"         : "GO:0005794",
	"MITOCHONDRIA"          : "GO:0005739",
	"CHLOROPLAST"           : "GO:0009507",
	"VACUOLE"               : "GO:0005773",
	"PEROXISOMES"           : "GO:0005777",
	"LYSOSOMES"             : "GO:0005764"
}

NUCLEUS_CELL_COMPONENTS = {
	"NUCLEAR_ENVELOPE" : "GO:0005635",
	"NUCLEOLUS"        : "GO:0005730"
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
def shuffle_data(labels1, labels2, abstracts, pmids_dataset):
    
    print("Shuffle dataset")
    
    labels1_shuffle = []
    labels2_shuffle = []
    abstracts_shuffle = []
    pmids_shuffle = []
    
    index_shuffle = np.arange(len(labels1))
    np.random.shuffle(index_shuffle)
    
    for i in index_shuffle:
        labels1_shuffle.append(labels1[i])
	labels2_shuffle.append(labels2[i])
        abstracts_shuffle.append(abstracts[i])
        pmids_shuffle.append(pmids_dataset[i])
    
    return (labels1_shuffle, labels2_shuffle, abstracts_shuffle, pmids_shuffle)



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

    class_names = sorted(list(PRIMARY_CELL_COMPONENTS.keys()))
    performance = metrics.classification_report(actual, predicted, target_names=class_names)
    print("\nPerformance:")
    print(performance)
    accuracy = metrics.accuracy_score(actual, predicted)
    print("\nAccuracy: ", accuracy*100)

    conf_matrix = metrics.confusion_matrix(actual, predicted)
    print("\nConfusion Matrix:")
    print(conf_matrix)
    
    f1 = metrics.precision_recall_fscore_support(actual, predicted)
    f1 = f1[2].mean(axis=0)
    
    return (accuracy, f1)


#display the classifier performance
def print_metrics_cytoplasm(actual, predicted):

    #class_names = sorted(list(CYTOPLASM_CELL_COMPONENTS.keys()))
    #class_names.append("OTHER")
    performance = metrics.classification_report(actual, predicted)
    print("\nPerformance:")
    print(performance)
    accuracy = metrics.accuracy_score(actual, predicted)
    print("\nAccuracy: ", accuracy*100)

    conf_matrix = metrics.confusion_matrix(actual, predicted)
    print("\nConfusion Matrix:")
    print(conf_matrix)

    f1 = metrics.precision_recall_fscore_support(actual, predicted)
    f1 = f1[2].mean(axis=0)

    return (accuracy, f1)


# remove duplicate papers/proteins appearing in both train and test
def remove_duplicate_papers(pmids_train, X_test, y1_test, y2_test, pmids_test):

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
        del y1_test[loc]
        del y2_test[loc]

    return X_test, y1_test, y2_test


def cytoplasm_train_and_test(X_train, y_train, X_test, y_test):

    print("\n******* CYTOPLASM CLASSIFIER *******")

    print("\n=====MULTINOMIAL NAIVE BAYES=====")
    t0 = time()
    classifier = MultinomialNB(alpha=.01)
    classifier.fit(X_train, y_train)
    predicted = classifier.predict(X_test)

    acc_mnb, f1_mnb = print_metrics_cytoplasm(y_test, predicted)
    print("\nTime to train (min.): ", (time()-t0)/60)


    print("\n========SVM========")
    t0 = time()
    classifier = svm.SVC()
    classifier.fit(X_train, y_train)
    predicted = classifier.predict(X_test)
	
    acc_svm, f1_svm = print_metrics_cytoplasm(y_test, predicted)
    print("\nTime to train (min.): ", (time()-t0)/60)



    print("\n=====ADABOOST=====")

    ada_real = AdaBoostClassifier(learning_rate=1,n_estimators=400,algorithm="SAMME.R")
    ada_real.fit(X_train, y_train)
    predicted = ada_real.predict(X_test)

    acc_ada, f1_ada = print_metrics_cytoplasm(y_test, predicted)
    print("\nTime to train (min.): ", (time()-t0)/60)



    print("\n=====RANDOM FOREST=====")
    t0 = time()
    classifier = RandomForestClassifier(n_estimators=400)
    classifier.fit(X_train, y_train)
    predicted = classifier.predict(X_test)
	
    acc_forest, f1_forest = print_metrics_cytoplasm(y_test, predicted)
    print("\nTime to train (min.): ", (time()-t0)/60)



#training and testing the main classifier
def train_and_test(X_train, y1_train, y2_train, X_test, y1_test, y2_test):
    
    print("Vectorizing features")
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5)
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)

    y1_train = np.array(y1_train)
    y2_train = np.array(y2_train)
    y1_test = np.array(y1_test)
    y2_test = np.array(y2_test)
	
    indexes = np.where(y1_train==PRIMARY_CELL_COMPONENTS_ID["CYTOPLASM"])[0]
    X_train_cytoplasm = X_train[indexes]
    y_train_cytoplasm = y2_train[indexes]


    print("\nn_samples: %d, n_features: %d" % X_train.shape)
    
    print("Train Set Counts:")
    counts = collections.Counter(y1_train)
    print("Total: ", len(y1_train))
    print(counts)
    
    print("Test Set Counts:")
    counts = collections.Counter(y1_test)
    print("Total: ", len(y1_test))
    print(counts)
    
    print("\n+++++++ PRIMARY CLASSIFIER +++++++")



    print("\n=====MULTINOMIAL NAIVE BAYES=====")
    t0 = time()
    classifier = MultinomialNB(alpha=.01)
    classifier.fit(X_train, y1_train)
    predicted = classifier.predict(X_test)
	
    acc_mnb, f1_mnb = print_metrics(y1_test, predicted)
	
    #re-classify the cytoplasm datapoints
    index1 = np.where(predicted==PRIMARY_CELL_COMPONENTS_ID["CYTOPLASM"])[0]
    index2 = np.where(y2_test==PRIMARY_CELL_COMPONENTS_ID["CYTOPLASM"])[0]
    indexes = np.intersect1d(index1, index2)
    X_test_cytoplasm = X_test[indexes]
    y_test_cytoplasm = y2_test[indexes]
    
    #print("\nCYTOPLASM CLASSIFICATION")
    #classifier2 = MultinomialNB(alpha=.01)
    #classifier2.fit(X_train_cytoplasm, y_train_cytoplasm)
    #predicted2 = classifier2.predict(X_test_cytoplasm)

    #acc_mnb, f1_mnb = print_metrics_cytoplasm(y_test_cytoplasm, predicted2)	
	

    print("\n========SVM========")
    t0 = time()
    classifier = svm.LinearSVC(multi_class='ovr')
    classifier.fit(X_train, y1_train)
    predicted = classifier.predict(X_test)
	
    acc_svm, f1_svm = print_metrics(y1_test, predicted)
    print("\nTime to train (min.): ", (time()-t0)/60)
	
    #re-classify the cytoplasm datapoints
    index1 = np.where(predicted==PRIMARY_CELL_COMPONENTS_ID["CYTOPLASM"])[0]
    index2 = np.where(y2_test==PRIMARY_CELL_COMPONENTS_ID["CYTOPLASM"])[0]
    indexes = np.intersect1d(index1, index2)
    X_test_cytoplasm = X_test[indexes]
    y_test_cytoplasm = y2_test[indexes]
    
    #print("\nCYTOPLASM CLASSIFICATION")
    #classifier2 = svm.LinearSVC(multi_class='ovr')
    #classifier2.fit(X_train_cytoplasm, y_train_cytoplasm)
    #predicted2 = classifier2.predict(X_test_cytoplasm)

    #acc_mnb, f1_mnb = print_metrics_cytoplasm(y_test_cytoplasm, predicted2)



    print("\n=====ADABOOST=====")
	

    classifier = svm.SVC(decision_function_shape='ovr')
    classifier.fit(X_train, y1_train)
    ada_real = AdaBoostClassifier(base_estimator=classifier,learning_rate=1,n_estimators=400,algorithm="SAMME")
    ada_real.fit(X_train, y1_train)
    predicted = ada_real.predict(X_test)

    acc_ada, f1_ada = print_metrics(y1_test, predicted)
    print("\nTime to train (min.): ", (time()-t0)/60)

    #re-classify the cytoplasm datapoints
    index1 = np.where(predicted==PRIMARY_CELL_COMPONENTS_ID["CYTOPLASM"])[0]
    index2 = np.where(y2_test==PRIMARY_CELL_COMPONENTS_ID["CYTOPLASM"])[0]
    indexes = np.intersect1d(index1, index2)
    X_test_cytoplasm = X_test[indexes]
    y_test_cytoplasm = y2_test[indexes]
  
    #print("\nCYTOPLASM CLASSIFICATION")
    #classifier2 = AdaBoostClassifier(learning_rate=1,n_estimators=400,algorithm="SAMME.R")
    #classifier2.fit(X_train_cytoplasm, y_train_cytoplasm)
    #predicted2 = classifier2.predict(X_test_cytoplasm)

    #acc_mnb, f1_mnb = print_metrics_cytoplasm(y_test_cytoplasm, predicted2)



    print("=====RANDOM FOREST=====")
    t0 = time()
    classifier = RandomForestClassifier(n_estimators=400)
    classifier.fit(X_train, y1_train)
    predicted = classifier.predict(X_test)
				
    acc_forest, f1_forest = print_metrics(y1_test, predicted)
    print("\nTime to train (min.): ", (time()-t0)/60)

    #re-classify the cytoplasm datapoints
    index1 = np.where(predicted==PRIMARY_CELL_COMPONENTS_ID["CYTOPLASM"])[0]
    index2 = np.where(y2_test==PRIMARY_CELL_COMPONENTS_ID["CYTOPLASM"])[0]
    indexes = np.intersect1d(index1, index2)
    X_test_cytoplasm = X_test[indexes]
    y_test_cytoplasm = y2_test[indexes]
  
    #print("\nCYTOPLASM CLASSIFICATION")
    #classifier2 = RandomForestClassifier(n_estimators=400)
    #classifier2.fit(X_train_cytoplasm, y_train_cytoplasm)
    #predicted2 = classifier2.predict(X_test_cytoplasm)

    #acc_mnb, f1_mnb = print_metrics_cytoplasm(y_test_cytoplasm, predicted2)



    return (acc_forest, f1_forest, acc_mnb, f1_mnb, acc_svm, f1_svm)


#K-fold validation (train and testing are called from here)
def k_fold_validation(K, labels1, labels2, abstracts, pmids_dataset):
    
	labels1_folds = np.array_split(labels1, K)
        labels2_folds = np.array_split(labels2, K)
    	abstracts_folds = np.array_split(abstracts, K)
    	pmids_dataset_folds = np.array_split(pmids_dataset, K)
    
	n_classifiers = 3
	accuracy_mat = np.empty([K, n_classifiers], dtype=float)
	f1_mat = np.empty([K, n_classifiers], dtype=float)
		
	fold = 1
	for fold in range(K):
	
		print("\n============ Fold " + str(fold) + " ==============\n")

        	k_fold_labels1 = np.delete(labels1_folds,fold,axis=0)
       		k_fold_labels1 = np.hstack(k_fold_labels1)
        	k_fold_labels1 = list(k_fold_labels1)
        
                k_fold_labels2 = np.delete(labels2_folds,fold,axis=0)
                k_fold_labels2 = np.hstack(k_fold_labels2)
                k_fold_labels2 = list(k_fold_labels2)

        	k_fold_abstracts = np.delete(abstracts_folds,fold,axis=0)
        	k_fold_abstracts = np.hstack(k_fold_abstracts)
        	k_fold_abstracts = list(k_fold_abstracts)
        
        	pmids_array = np.delete(pmids_dataset_folds,fold,axis=0)
        	k_fold_pmids = np.hstack(pmids_array)
        	k_fold_pmids = list(k_fold_pmids)
        
        	X_train = k_fold_abstracts
        	y1_train = k_fold_labels1
		y2_train = k_fold_labels2
        	pmids_train = k_fold_pmids
        
        	X_test = list(abstracts_folds[fold])
        	y1_test = list(labels1_folds[fold])
		y2_test = list(labels2_folds[fold])
        	pmids_test = list(pmids_dataset_folds[fold])
        
		#remove duplicates from the test set
		(X_test, y1_test, y2_test) = remove_duplicate_papers(pmids_train, X_test, y1_test, y2_test, pmids_test)
        
		### TRAINING ANG TESTING ###
		(acc1, f1_1,  acc2, f1_2, acc3, f1_3) = train_and_test(X_train, y1_train, y2_train, X_test, y1_test, y2_test)
        
		accuracy_mat[fold][0] = acc1*100
		accuracy_mat[fold][1] = acc2*100
		accuracy_mat[fold][2] = acc3*100

		f1_mat[fold][0] = f1_1*100
		f1_mat[fold][1] = f1_2*100
		f1_mat[fold][2] = f1_3*100


	avg_acc = accuracy_mat.mean(axis=0)
	avg_f1 = f1_mat.mean(axis=0)


	print("\n=============== SUMMARY REPORT ===============")
    
	print("\nAverage Accuracy:\n")
	print("Random Forest: ", avg_acc[0])
	#print("AdaBoost (w/ SVM): ", avg_acc[1])
	print("Multinomial NB: ", avg_acc[1])
	print("SVM: ", avg_acc[2])
    
	print("\nAverage F1-score:\n")
	print("Random Forest: ", avg_f1[0])
	#print("AdaBoost (w/ SVM): ", avg_f1[1])
	print("Multinomial NB: ", avg_f1[1])
	print("SVM: ", avg_f1[2])
    
	print("\nTotal time duration: ", (time() - time_start)/60)


################################################################


print("Getting GO and PubMed records from file")

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


## get the cell components descendants ##
# structure of the dict:
# descendant_dict["NUCLEUS"] = ['GO:0000', 'GO:0001', etc]

primary_descendant_dict = {}
primary_components = sorted(list(PRIMARY_CELL_COMPONENTS.keys()))
for cell_comp in primary_components:
    primary_descendant_dict[cell_comp] = get_descendants(PRIMARY_CELL_COMPONENTS[cell_comp])


cytoplasm_descendant_dict = {}
cytoplasm_components = sorted(list(CYTOPLASM_CELL_COMPONENTS.keys()))
for cell_comp in cytoplasm_components:
    cytoplasm_descendant_dict[cell_comp] = get_descendants(CYTOPLASM_CELL_COMPONENTS[cell_comp])

cytoplasm_count = len(cytoplasm_components)

print("Getting abstracts and assigning labels")

primary_labels = list()
cytoplasm_labels = list()
abstracts = list()
pmids_dataset = list() #every datapoint will have an associated pubmed id in this list



for pmid in pmids:
    matching_proteins = data[data[:,2]==pmid]
    go_terms = list(set(matching_proteins[:,1]))
    matching_pub = data2[data2[:,1]==pmid]
    text = matching_pub[0][4]
    text = text_preprocessing(text)
    for term in go_terms:
        for i in range(len(primary_components)):
            if term in primary_descendant_dict[primary_components[i]]:
                primary_labels.append(i)
                abstracts.append(text)
                pmids_dataset.append(pmid)
                if primary_components[i] == "CYTOPLASM":
                    cytoplasm_labels.append(cytoplasm_count)
                    for i in range(len(cytoplasm_components)):
                        if term in cytoplasm_descendant_dict[cytoplasm_components[i]]:
                            cytoplasm_labels.append(i)
                else:
                    cytoplasm_labels.append(-1)


#shuffle dataset
(primary_labels, cytoplasm_labels, abstracts, pmids_dataset) = shuffle_data(primary_labels, cytoplasm_labels,abstracts, pmids_dataset)

#k-fold validation
#K=5 #K-fold validation
#k_fold_validation(K, primary_labels, cytoplasm_labels, abstracts, pmids_dataset)

half = (len(primary_labels)/5)*4
X_train = abstracts[:half]
y1_train = primary_labels[:half]
y2_train = cytoplasm_labels[:half]
pmids_train = pmids_dataset[:half]

X_test = abstracts[half:]
y1_test = primary_labels[half:]
y2_test = cytoplasm_labels[half:]
pmids_test = pmids_dataset[half:]

(X_test, y1_test, y2_test) = remove_duplicate_papers(pmids_train, X_test, y1_test, y2_test, pmids_test)
(acc1, f1_1,  acc2, f1_2, acc3, f1_3) = train_and_test(X_train, y1_train, y2_train, X_test, y1_test, y2_test)

print("DONE!\n")




