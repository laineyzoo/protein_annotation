from __future__ import division
from __future__ import print_function

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
from sklearn import cluster

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

GO_BIOLOGICAL_PROCESS = {

    "REPRODUCTION" 		: "GO:0000003",
    "CELL_KILLING" 		: "GO:0001906",
    "IMMUNE_SYSTEM_PROCESS" 	: "GO:0002376",
    "BEHAVIOR"			: "GO:0007610",
    "METABOLIC_PROCESS" 	: "GO:0008152",
    "CELLULAR_PROCESS" 		: "GO:0009987",
    "REPRODUCTIVE_PROCESS" 	: "GO:0022414",
    "BIOLOGICAL_ADHESION" 	: "GO:0022610",
    "SIGNALING"			: "GO:0023052",
    "MULTICELL_ORGANISM_PROCESS": "GO:0032501",
    "DEVELOPMENTAL_PROCESS" 	: "GO:0032502",
    "GROWTH" 			: "GO:0040007",
    "LOCOMOTION" 		: "GO:0040011",
    "SINGLE_ORGANISM_PROCESS" 	: "GO:0044699",
    "BIOLOGICAL_PHASE" 		: "GO:0044848",
    "RHYTHMIC_PROCESS" 		: "GO:0048511",
    "RESPONSE_STIMULUS" 	: "GO:0050896",
    "LOCALIZATION" 		: "GO:0051179",
    "MULTI_ORGANISM_PROCESS" 	: "GO:0051704",
    "BIOLOGICAL_REGULATION" 	: "GO:0065007",
    "BIOGENESIS" 		: "GO:0071840",
    "CELL_AGGREGATION" 		: "GO:0098743",
    "DETOXIFICATION" 		: "GO:0098754",
    "SYNAPTIC_TRANSMISSION" 	: "GO:0099531"

}



GO_MOLECULAR_FUNCTION = {
    "PROTEIN_BINDING" 		: "GO:0000988",
    "NUCLEIC_ACID_BINDING" 	: "GO:0001071",
    "CATALYTIC" 		: "GO:0003824",
    "SIGNAL_TRANSDUCER" 	: "GO:0004871",
    "STRUCTURAL_MOLECULE" 	: "GO:0005198",
    "TRANSPORTER" 		: "GO:0005215",
    "BINDING" 			: "GO:0005488",
    "ELECTRON_CARRIER" 		: "GO:0009055",
    "MORPHOGEN" 		: "GO:0016015",
    "ANTIOXIDANT" 		: "GO:0016209",
    "METALLOCHAPERONE" 		: "GO:0016530",
    "PROTEIN_TAG" 		: "GO:0031386",
    "D-ALANYL_CARRIER" 		: "GO:0036370",
    "CHEMOATTRACTANT" 		: "GO:0042056",
    "TRANSLATION_REGULATOR" 	: "GO:0045182",
    "CHEMOREPELLENT" 		: "GO:0045499",
    "NUTRIENT_RESERVOIR" 	: "GO:0045735",
    "MOLECULAR_TRANSDUCER" 	: "GO:0060089",
    "MOLECULAR_FUNCTION_REGULATOR" : "GO:0098772"
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

    performance = metrics.classification_report(actual, predicted, target_names=cell_components)
    print("\nPerformance:")
    print(performance)

    accuracy = metrics.accuracy_score(actual, predicted)
    print("Accuracy: ", accuracy*100)

    conf_matrix = metrics.confusion_matrix(actual, predicted)
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print(conf_matrix)    

    f1 = metrics.precision_recall_fscore_support(actual, predicted)
    f1 = f1[2].mean(axis=0)
    
    return (accuracy, f1)


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



#training and testing phase with 4 types of classifiers
def train_and_test(X_train, y_train, X_test, y_test):
    
    print("Vectorizing features")
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5)
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)
    
    print("\nn_samples: %d, n_features: %d" % X_train.shape)
    
    
    print("\nTrain Set Counts:")
    counts = collections.Counter(y_train)
    print("\nTotal: ", len(y_train))
    print(counts)
    
    print("\nTest Set Counts:")
    counts = collections.Counter(y_test)
    print("\nTotal: ", len(y_test))
    print(counts)
    
    print("\nTraining and Testing")
	
    print("\n=====MULTINOMIAL NAIVE BAYES=====")
    t0 = time()
    classifier = MultinomialNB(alpha=.01)
    classifier.fit(X_train, y_train)
    predicted = classifier.predict(X_test)
		
    acc_mnb, f1_mnb = print_metrics(y_test, predicted)
    
    
    print("\n========SVM========")
    t0 = time()
    classifier = svm.LinearSVC(multi_class='ovr')
    classifier.fit(X_train, y_train)
    predicted = classifier.predict(X_test)
	
    acc_svm, f1_svm = print_metrics(y_test, predicted)


   
    print("\n=====RANDOM FOREST=====")
    t0 = time()
    classifier = RandomForestClassifier(n_estimators=400, n_jobs=10)
    classifier.fit(X_train, y_train)
    predicted = classifier.predict(X_test)
    
    acc_rf, f1_rf = print_metrics(y_test, predicted)
    

    return (acc_mnb, f1_mnb, acc_svm, f1_svm, acc_rf, f1_rf)



#K-fold validation (train and testing are called from here)
def k_fold_validation(K, labels, abstracts, pmids_dataset):
    
    labels_folds = np.array_split(labels, K)
    abstracts_folds = np.array_split(abstracts, K)
    pmids_dataset_folds = np.array_split(pmids_dataset, K)
    
    n_classifiers = 3
    accuracy_mat = np.empty([K, n_classifiers], dtype=float)
    f1_mat = np.empty([K, n_classifiers], dtype=float)
    
    for fold in range(K):
        
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
        (acc1, f1_1,  acc2, f1_2, acc3, f1_3) = train_and_test(k_fold_abstracts, k_fold_labels, X_test, y_test)
        
        accuracy_mat[fold][0] = acc1*100
        accuracy_mat[fold][1] = acc2*100
        accuracy_mat[fold][2] = acc3*100
        #accuracy_mat[fold][3] = acc4*100
        f1_mat[fold][0] = f1_1*100
        f1_mat[fold][1] = f1_2*100
        f1_mat[fold][2] = f1_3*100
        #f1_mat[fold][3] = f1_4*100

    avg_acc = accuracy_mat.mean(axis=0)
    avg_f1 = f1_mat.mean(axis=0)

    print ("\n=============== SUMMARY REPORT ===============")
    
    print("\nAverage Accuracy:\n")
    print("MNB: ", avg_acc[0])
    print("SVM: ", avg_acc[1])
    print("Random Forest: ", avg_acc[2])
    #print("SVM: ", avg_acc[3])
    
    print("\nAverage F1-score:\n")
    print("MNB: ", avg_f1[0])
    print("SVM: ", avg_f1[1])
    print("Random Forest: ", avg_f1[2])
    #print("SVM: ", avg_f1[3])
    
    print("\nTotal time duration: ", (time() - time_start)/60)


#########################################################################################


print("\nSTART\n")
print("Getting GO and PubMed records from file")

time_start = time()
file1 = open("protein_records.csv","r")
reader = csv.reader(file1)
data = np.array(list(reader))
data_cc = data[data[:,4]=="C"]
data_bp = data[data[:,4]=="P"]
data_mf = data[data[:,4]=="F"]

file2 = open("pubmed_records.csv","r")
reader = csv.reader(file2)
data2 = np.array(list(reader))

proteins_cc = list(set(data_cc[:,0]))
pmids_cc = list(set(data_cc[:,2]))

proteins_bp = list(set(data_bp[:,0]))
pmids_bp = list(set(data_bp[:,2]))

proteins_mf = list(set(data_mf[:,0]))
pmids_mf = list(set(data_mf[:,2]))

file1.close()


## get the cell components descendants ##
# structure of the dict:
# descendant_dict["NUCLEUS"] = ['GO:0000', 'GO:0001', etc]

descendant_dict_cc = {}
cell_components = sorted(list(GO_CELL_COMPONENTS.keys()))
print("Cell components: ", len(cell_components))
for cell_comp in cell_components:
    descendant_dict_cc[cell_comp] = get_descendants(GO_CELL_COMPONENTS[cell_comp])

#descendant_dict_bp = {}
#biological_process = sorted(list(GO_BIOLOGICAL_PROCESS.keys()))
#print("Biological process: ", len(biological_process))
#for bio_process in biological_process:
#    descendant_dict_bp[bio_process] = get_descendants(GO_BIOLOGICAL_PROCESS[bio_process])

descendant_dict_mf = {}
molecular_function = sorted(list(GO_MOLECULAR_FUNCTION.keys()))
print("Molecular function: ", len(molecular_function))
for molecule_func in molecular_function:
    descendant_dict_mf[molecule_func] = get_descendants(GO_MOLECULAR_FUNCTION[molecule_func])


file1.close()


print("Getting abstracts")

print("\nCellular Components")
labels_cc = list()
abstracts_cc = list()
pmids_dataset_cc = list() #every datapoint will have an associated pubmed id in this list

for pmid in pmids_cc:

    matching_proteins = data_cc[data_cc[:,2]==pmid]
    go_terms = list(set(matching_proteins[:,1]))
    matching_pub = data2[data2[:,1]==pmid]
    text = matching_pub[0][4]
    text = text_preprocessing(text)
    
    for term in go_terms:
        for i in range(len(cell_components)):
            if term in descendant_dict_cc[cell_components[i]]:
                labels_cc.append(i)
                abstracts_cc.append(text)
                pmids_dataset_cc.append(pmid)
    


(labels_cc, abstracts_cc, pmids_dataset_cc) = shuffle_data(labels_cc, abstracts_cc, pmids_dataset_cc)

vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5)
X_cc = vectorizer.fit_transform(abstracts_cc)

#print("\nBiological process")

#labels_bp = list()
#abstracts_bp = list()
#pmids_dataset_bp = list() #every datapoint will have an associated pubmed id in this list

#for pmid in pmids_bp:

#    matching_proteins = data_bp[data_bp[:,2]==pmid]
#    go_terms = list(set(matching_proteins[:,1]))
#    matching_pub = data2[data2[:,1]==pmid]
#    text = matching_pub[0][4]
#    text = text_preprocessing(text)

#    for term in go_terms:
#        for i in range(len(biological_process)):
#            if term in descendant_dict_bp[biological_process[i]]:
#                labels_bp.append(i)
#                abstracts_bp.append(text)
#                pmids_dataset_bp.append(pmid)

#(labels_bp, abstracts_bp, pmids_dataset_bp) = shuffle_data(labels_bp, abstracts_bp, pmids_dataset_bp)

#vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5)
#X_bp = vectorizer.fit_transform(abstracts_bp)

print("\nMolecular function")

labels_mf = list()
abstracts_mf = list()
pmids_dataset_mf = list() #every datapoint will have an associated pubmed id in this list

for pmid in pmids_mf:

    matching_proteins = data_mf[data_mf[:,2]==pmid]
    go_terms = list(set(matching_proteins[:,1]))
    matching_pub = data2[data2[:,1]==pmid]
    text = matching_pub[0][4]
    text = text_preprocessing(text)

    for term in go_terms:
        for i in range(len(molecular_function)):
            if term in descendant_dict_mf[molecular_function[i]]:
                labels_mf.append(i)
                abstracts_mf.append(text)
                pmids_dataset_mf.append(pmid)



(labels_mf, abstracts_mf, pmids_dataset_mf) = shuffle_data(labels_mf, abstracts_mf, pmids_dataset_mf)

vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5)
X_mf = vectorizer.fit_transform(abstracts_mf)


iter = 5
score_mat = np.empty([iter, len(range(10,21))], dtype=float)
print("\nSpectral Clustering Cellular Components")
for i in range(iter):
	for k in range(10,21):
   		spectral = cluster.SpectralClustering(n_clusters=k, eigen_solver='arpack', affinity='cosine')
    		spectral.fit(X_cc)
    		cluster_labels = spectral.labels_
    		score = metrics.silhouette_score(X_cc, cluster_labels, metric='cosine')
		score_mat[i,k-10] = score


mean_scores = score_mat.mean(axis=0)
print("Mean scores:\n", mean_scores)


#score_mat = np.empty([iter, len(range(20,30))], dtype=float)
#print("\nClustering Biological Processes")
#for i in range(iter):
#        for k in range(20,30):
#                spectral = cluster.SpectralClustering(n_clusters=k, eigen_solver='arpack', affinity='cosine')
#                spectral.fit(X_bp)
#                cluster_labels = spectral.labels_
#                score = metrics.silhouette_score(X_bp, cluster_labels, metric='cosine')
#                score_mat[i,k-20] = score


#mean_scores = score_mat.mean(axis=0)
#print("Mean scores:\n", mean_scores)



score_mat = np.empty([iter, len(range(15,25))], dtype=float)
print("\nSpectral Clustering Molecular Function")
for i in range(iter):
        for k in range(15,25):
                spectral = cluster.SpectralClustering(n_clusters=k, eigen_solver='arpack', affinity='cosine')
                spectral.fit(X_mf)
                cluster_labels = spectral.labels_
                score = metrics.silhouette_score(X_mf, cluster_labels, metric='cosine')
                score_mat[i,k-15] = score


mean_scores = score_mat.mean(axis=0)
print("Mean scores:\n", mean_scores)



print("\nDONE!\n")





