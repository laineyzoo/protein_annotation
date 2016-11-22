from __future__ import division
from __future__ import print_function

import numpy as np
import collections
import json
import string
import re
import sys, getopt
from time import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm



## shuffle the dataset
def shuffle_data(sequence, sequence_ids, classification):
	sequence_shuffle = []
	sequence_ids_shuffle = []
	classes_shuffle = []
	index_shuffle = np.arange(len(sequence))
	np.random.shuffle(index_shuffle)
	for i in index_shuffle:
		sequence_shuffle.append(sequence[i])
		sequence_ids_shuffle.append(sequence_ids[i])
		classes_shuffle.append(classification[i])
	return (sequence_shuffle, sequence_ids_shuffle, classes_shuffle)


#divide a sequence into kmers
def create_kmers(sequence, k):
	if k<len(sequence):
		kmers = list()
		for i in range(len(sequence)+1-k):
			kmers.append(sequence[i:i+k])
		seq = " ".join(k for k in kmers)
		return seq
	else:
		return sequence


#check if all elements of an array are equal
def array_equal(arr):
	return all(np.isclose(x, arr[0]) for x in arr)


#get distance to root
def path_to_root(rank, taxonomy):
	if root==rank:
		return 0
	q = collections.deque()
	q.append(rank)
	dist = 0
	taxonomy[root]["parent"] = []
	while q:
		node = q.popleft()
		dist += 1
		parents = taxonomy[node]["parent"]
		q.extend(parents)
	return (dist-1)


#return LCA of a list of nodes
def get_lca(node_list, taxonomy):
	if len(node_list)==0:
		return ""
	elif len(node_list)==1:
		return node_list[0]
	paths = list()
	[paths.append(propagate_labels([node], taxonomy)) for node in node_list]
	intersect = list()
	path0 = paths[0]
	path1 = paths[1]
	intersect = list(set(path0) & set(path1))
	if len(node_list)>2:
		for i in range(2, len(paths)):
			intersect = list(set(intersect) & set(paths[i]))
	intersect_len = [path_to_root(inter, taxonomy) for inter in intersect]
	lca = intersect[intersect_len.index(max(intersect_len))]
	return lca

def filter_prediction(pred_labels, idx, taxonomy):
	pred = pred_labels
	path_len = [path_to_root(j, taxonomy) for j in pred]
	counts = collections.Counter(path_len)
	values = counts.values()
	error_type = 0
	#check if we have an ambiguous prediction
	if max(values)>1:
		lowest_pred_len = max(path_len)
		prob_ont = prob_dict[sequence_ids_test[idx]]
		#check if Case 1 vs 2 & 3
		if counts[lowest_pred_len]>1:
			lowest_labels = list()
			lowest_labels_prob = list()
			for i in range(len(path_len)):
				if path_len[i]==lowest_pred_len:
					lowest_labels.append(pred[i])
					lowest_labels_prob.append(prob_ont[pred[i]])
			#check for Case 3: ambiguous prediction is at the lowest level and 1 or more higher levels
			if sum((np.array(values)>1)*1)>1:
				error_type = 3
				if array_equal(lowest_labels_prob):
					pred,error_type = filter_prediction(lowest_labels, idx,taxonomy)
				else:
					highest_prob_label = lowest_labels[lowest_labels_prob.index(max(lowest_labels_prob))]
					pred = propagate_labels([highest_prob_label], taxonomy)
			else:
				#Case 2: ambiguous prediction is at the lowest level only
				error_type = 2
				#Case 2 solution (check for Case 4)
				#if all prob are equal, return LCA
				if array_equal(lowest_labels_prob):
					#we just need to return LCA since it will just get propagated upwards later
					lca = get_lca(lowest_labels, taxonomy)
					pred = propagate_labels([lca], taxonomy)
				#one prob is higher than the rest, remove the rest
				else:
					highest_prob_label = lowest_labels[lowest_labels_prob.index(max(lowest_labels_prob))]
					pred = propagate_labels([highest_prob_label], taxonomy)
		#Case 1
		else:
			#Case 1: ambiguous prediction is at a higher level but not in the lowest level
			error_type = 1
			max_val = max(values)
			ambiguous_level = 0
			for k in counts.keys():
				if counts[k]==max_val:
					ambiguous_level = k
			ambiguous_labels = list()
			ambiguous_labels_prob = list()
			for i in range(len(path_len)):
				if path_len[i]==ambiguous_level:
					ambiguous_labels.append(pred[i])
					ambiguous_labels_prob.append(prob_ont[pred[i]])
			lowest_label = pred_labels[path_len.index(lowest_pred_len)]
			longest_path = propagate_labels([lowest_label], taxonomy)
			highest_prob_label = ambiguous_labels[ambiguous_labels_prob.index(max(ambiguous_labels_prob))]
			#check if probs at the ambiguous level are equal
			if array_equal(ambiguous_labels_prob) or (highest_prob_label in longest_path):
				pred = longest_path
			else:
				#one sibling has higher prob, return it
				pred = propagate_labels([highest_prob_label], taxonomy)
	else:
		error_type=0
	return pred,error_type



#compute leaf-level accuracy
def compute_accuracy(true_labels, pred_labels, taxonomy):
	acc = 0
	pred_path_len = [path_to_root(x, taxonomy) for x in pred_labels]
	true_path_len = [path_to_root(x, taxonomy) for x in true_labels]
	if max(true_path_len) == max(pred_path_len):
		max_path = max(true_path_len)
		true_leaf = true_labels[true_path_len.index(max_path)]
		if true_leaf in pred_labels:
			acc = 1
	else:
		acc = -1
	return acc


#compute precision, recall and accuracy for one test point
def evaluate_prediction(true_labels, pred_labels):
	tp = set(pred_labels).intersection(true_labels)
	precision = 0.0
	recall = 0.0
	if len(pred_labels)!=0:
		precision = len(tp)/len(pred_labels)
	if len(true_labels)!=0:
		recall = len(tp)/len(true_labels)
	return precision,recall


def propagate_labels(labels, taxonomy):
	label_list = []
	for label in labels:
		labels_prop = list()
		labels_prop.append(label)
		q = collections.deque()
		q.append(label)
		#traverse ontology upwards from node to root
		taxonomy[root]["parent"] = []
		while len(q)>0:
			node = q.popleft()
			parents = taxonomy[node]["parent"]
			labels_prop.extend(parents)
			q.extend(parents)
		#remove duplicates in the label set
		labels_prop = list(set(labels_prop))
		#add this label set to our list
		label_list.extend(labels_prop)
	label_list = list(set(label_list))
	return label_list


#give prediction probabilities for this test point
def predict_class(test_point):
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



######################################  MAIN  ############################################


if __name__ == "__main__":
	if len(sys.argv[1:]) < 8:
		print("This script requires 8 arguments: root, kmer size, sample_threshold, classifier, dataset, filtering, no. of folds, save_results")
	else:
		time_start_all = time()
		print("\n=====START N-fold Cross-validation=====\n")
		print("Settings: ")

		root = sys.argv[1]
		print("Root: ", root)

		kmer = int(sys.argv[2])
		print("K-mer size: ", kmer)
	
		sample_threshold = int(sys.argv[3])
		print("Sample threshold: ", sample_threshold)
	
		algo = sys.argv[4]
		if algo == "S":
			print("Classifier: SVM")
		else:
			print("Classifier: Naive-Bayes")
		
		dataset = sys.argv[5]
		if dataset == "R":
			print("Dataset: RDP")
			f1 = open("rdp_taxonomy.json","r") 
			f2 = open("rdp_data_dict.json","r")
		else:
			print("Dataset: SILVA")
			f1 = open("taxonomy.json", "r") 
			f2 = open("silva_dict.json", "r")

		filtering = sys.argv[6]
		print("Filtering: ", filtering)

		folds = int(sys.argv[7])
		print("Folds: ", folds)

		save_result = sys.argv[8]
		print("Save Results? ", save_result)
	
		taxonomy = json.load(f1)
		data_dict = json.load(f2)
		f1.close()
		f2.close()

		sequence_ids = list()
		sequence = list()
		classification = list()

		#from this point, all nodes above the root will not be considered
		taxonomy[root]["parent"] = []
	
		#Only use the parts of the dataset that belongs to the root
		desc_root = taxonomy[root]["descendants"]

		keys = data_dict.keys()
		for key in keys:
			if data_dict[key]["class"][-1] in desc_root:
				sequence_ids.append(key)
				seq = create_kmers(data_dict[key]["sequence"],kmer)
				sequence.append(seq)
				classification.append(data_dict[key]["class"][-1])

		#shuffle dataset
		print("\nPreparing the dataset")
		(sequence, sequence_ids, classification) = shuffle_data(sequence, sequence_ids, classification)
		
		#divide the dataset into n-folds
		total = len(sequence)
		div = int(total/folds)
		precision_kfold = []
		recall_kfold = []
		acc_kfold = []
		f1_kfold = []
		thresh_kfold = []
		thresh_acc_kfold = []
		train_size_kfold = []
		test_size_kfold = []
		if filtering == "Y":
			precision_filter_kfold = []
			recall_filter_kfold = []
			acc_filter_kfold = []
			f1_filter_kfold = []
			thresh_filter_kfold = []
			thresh_acc_filter_kfold = []
			error_types_kfold = []
		for fold in range(folds):
			print("Fold ", (fold+1))
			test_start = int(fold*div)
			test_end = int((fold+1)*div)
			
			#prepare train set
			if test_start == 0:
				train_start = div
				train_end = total
				X_train = sequence[train_start:train_end]
				classification_train = classification[train_start:train_end]
				sequence_ids_train = sequence_ids[train_start:train_end]
			else:
				train_start = 0
				train_end = test_start
				if test_end == total:
					X_train = sequence[train_start:train_end]
					classification_train = classification[train_start:train_end]
					sequence_ids_train = sequence_ids[train_start:train_end]
				else:
					train_start2 = test_end
					train_end2 = total
					X_train = sequence[train_start:train_end]+sequence[train_start2:train_end2]
					classification_train = classification[train_start:train_end]+classification[train_start2:train_end2]
					sequence_ids_train = sequence_ids[train_start:train_end]+sequence_ids[train_start2:train_end2]

			#prepare test set
			if dataset=="R":
				X_test = []
				classification_test = []
				sequence_ids_test = []
				
				X_test_temp = sequence[test_start:test_end]
				classification_test_temp = classification[test_start:test_end]
				sequence_ids_test_temp = sequence_ids[test_start:test_end]

				for i in range(len(X_test_temp)):
					if classification_train.count(classification_test_temp[i])>=2:
						X_test.append(X_test_temp[i])
						sequence_ids_test.append(sequence_ids_test_temp[i])
						classification_test.append(classification_test_temp[i])
			else:
				X_test = sequence[test_start:test_end]
				classification_test = classification[test_start:test_end]
				sequence_ids_test = sequence_ids[test_start:test_end]

			train_size = len(X_train)
			test_size = len(X_test)
			train_size_kfold.append(train_size)
			test_size_kfold.append(test_size)
			print("Train set: ", train_size)
			print("Test set: ", test_size)
			#vectorize features
			vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5)
			X_train = vectorizer.fit_transform(X_train)
			X_test = vectorizer.transform(X_test)

			#create binary classifiers
			print("Creating classifiers")
			node_count = 0
			classifiers = {}
			positive_count = {}
			for desc in desc_root:
				descendants = taxonomy[desc]["descendants"]
				node_count+=1
				y_train = list()
				for c in classification_train:
					if c in descendants:
						y_train.append(1)
					else:
						y_train.append(0)
				if y_train.count(1)>=sample_threshold:
					if algo != "S" or y_train.count(1)==len(y_train):
						clf = MultinomialNB(alpha=.01).fit(X_train, y_train)
					else:
						clf = svm.SVC(probability=True)
						clf.fit(X_train,y_train)
					classifiers[desc] = clf
					positive_count[desc] = y_train.count(1)
			print("Done creating classifiers. No. of classifiers: ", len(classifiers))

			#run classifiers on the test set
			print("Running classifiers on the test set")
			prob_dict = {}
			for i in range(len(sequence_ids_test)):
				test_point = X_test[i]
				prob_dict[sequence_ids_test[i]] = predict_class(test_point)

			#plot the precision/recall/f1 vs threshold
			print("Calculating F1/recall/precision by threshold")
			precision_list = []
			recall_list = []
			acc_list = []
			f1_list = []
			results_dict = {}
			if filtering == "Y":
				precision_filter_list = []
				recall_filter_list = []	
				acc_filter_list = []	
				f1_filter_list = []			
				error_types = []
			for thresh in range(0,101):
				thresh = float(thresh)/100
				total_precision = 0
				total_recall = 0
				total_acc = 0
				total_pred = 0
				if filtering == "Y":
					total_precision_filter = 0
					total_recall_filter = 0
					total_acc_filter = 0
					total_pred_filter = 0
					case_0 = 0
					case_1 = 0
				for i in range(len(sequence_ids_test)):
					true_labels = propagate_labels([classification_test[i]], taxonomy)
					prob_ontology = prob_dict[sequence_ids_test[i]]
					positive_labels = list()
					for key in classifiers.keys():
						if prob_ontology[key] >= thresh:
							positive_labels.append(key)
					filtered_labels = []
					if filtering == "Y":
						filtered_labels,case_type = filter_prediction(positive_labels,i, taxonomy)
						if case_type == 0:
							case_0+=1
						elif case_type == 1:
							case_1+=1	
						precision_filter,recall_filter = evaluate_prediction(true_labels, filtered_labels)
						acc_filter = compute_accuracy(true_labels, filtered_labels, taxonomy)
						total_precision_filter+=precision_filter
						total_recall_filter+=recall_filter
						if acc_filter>=0:
							total_acc_filter+=acc_filter
							total_pred_filter+=1
					predicted_labels = propagate_labels(positive_labels, taxonomy)
					precision,recall = evaluate_prediction(true_labels, predicted_labels)
					acc = compute_accuracy(true_labels, predicted_labels, taxonomy)
					total_precision+=precision
					total_recall+=recall
					if acc>=0:
						total_acc+=acc
						total_pred+=1
					results_dict[sequence_ids_test[i]]={}
					results_dict[sequence_ids_test[i]]["true"] = true_labels
					results_dict[sequence_ids_test[i]]["predicted"] = predicted_labels
					if filtering == "Y":
						results_dict[sequence_ids_test[i]]["filtered"] = filtered_labels

				if filtering == "Y":
					final_precision_filter = total_precision_filter/len(sequence_ids_test)
					final_recall_filter = total_recall_filter/len(sequence_ids_test)
					if total_pred_filter>0:
						final_acc_filter = total_acc_filter/total_pred_filter
					else:
						final_acc_filter = 0
					final_f1_filter = (2*final_precision_filter*final_recall_filter)/(final_precision_filter+final_recall_filter)
					precision_filter_list.append(final_precision_filter)
					recall_filter_list.append(final_recall_filter)
					acc_filter_list.append(final_acc_filter)
					f1_filter_list.append(final_f1_filter)
					error_types.append([case_0, case_1])

				final_precision = total_precision/len(sequence_ids_test)
				final_recall = total_recall/len(sequence_ids_test)
				if total_pred>0:
					final_acc = total_acc/total_pred
				else:
					final_acc = 0
				final_f1 = (2*final_precision*final_recall)/(final_precision+final_recall)
				precision_list.append(final_precision)
				recall_list.append(final_recall)
				acc_list.append(final_acc)
				f1_list.append(final_f1)
				
				if thresh > 0.10 and save_result=="Y":
					with open("results/results_dict_"+root+"_"+str(kmer)+"_"+str(thresh)+"_"+str(fold)+"_"+dataset+".json","w") as f:
						json.dump(results_dict, f)

			if filtering == "Y":
				max_f1_filter = max(f1_filter_list)
				max_thresh_filter = f1_filter_list.index(max_f1_filter)
				max_precision_filter = precision_filter_list[max_thresh_filter]
				max_recall_filter = recall_filter_list[max_thresh_filter]
				max_acc_filter_2 = acc_filter_list[max_thresh_filter]
				max_acc_filter = max(acc_filter_list)
				max_acc_thresh_filter = acc_filter_list.index(max_acc_filter)
				max_type_0 = error_types[max_thresh_filter][0]
				max_type_1 = error_types[max_thresh_filter][1]
				max_type_n = len(sequence_ids_test)-(max_type_1+max_type_0)	
				f1_filter_kfold.append(max_f1_filter)
				precision_filter_kfold.append(max_precision_filter)
				recall_filter_kfold.append(max_recall_filter)
				acc_filter_kfold.append((max_acc_filter, max_acc_filter_2))		
				thresh_filter_kfold.append(max_thresh_filter)
				thresh_acc_filter_kfold.append(max_acc_thresh_filter)
				error_types_kfold.append([max_type_0, max_type_1, max_type_n])
			max_f1 = max(f1_list)
			max_thresh = f1_list.index(max_f1)
			max_precision = precision_list[max_thresh]
			max_recall = recall_list[max_thresh]
			max_acc_2 = acc_list[max_thresh]
			max_acc = max(acc_list)
			max_acc_thresh = acc_list.index(max_acc)
			f1_kfold.append(max_f1)
			precision_kfold.append(max_precision)
			recall_kfold.append(max_recall)
			acc_kfold.append((max_acc, max_acc_2))
			thresh_kfold.append(max_thresh)
			thresh_acc_kfold.append(max_acc_thresh)

		print("\n-----Results-----")
		print("Avg F1: ", np.mean(np.array(f1_kfold)))
		print("Avg Precision: ", np.mean(np.array(precision_kfold)))
		print("Avg Recall: ", np.mean(np.array(recall_kfold)))
		print("Avg Accuracy: ", np.mean(np.array(acc_kfold)[:,1]))
		print("Avg Threshold: ", np.mean(np.array(thresh_kfold)))
		print("Best Accuracy: ", np.mean(np.array(acc_kfold)[:,0]))
		print("Best Accuracy Thresh: ", np.mean(np.array(thresh_acc_kfold)))
		
		if filtering == "Y":
			print("\nAvg Filtered F1: ", np.mean(np.array(f1_filter_kfold)))
			print("Avg Filtered Precision: ", np.mean(np.array(precision_filter_kfold)))
			print("Avg Filtered Recall: ", np.mean(np.array(recall_filter_kfold)))
			print("Avg Filtered Accuracy: ", np.mean(np.array(acc_filter_kfold)[:,1]))
			print("Avg Filtered Threshold: ", np.mean(np.array(thresh_filter_kfold)))
			print("Best Filtered Accuracy: ", np.mean(np.array(acc_filter_kfold)[:,0]))
			print("Best Filtered Accuracy Thresh: ", np.mean(np.array(thresh_acc_filter_kfold)))
			means = np.mean(np.array(error_types_kfold), axis=0)
			print("\nAvg Case 0 Predictions: ", means[0])
			print("Avg Case 1 Predictions: ", means[1])
			print("Avg Case 2 or 3 Predictions: ", means[2])

		for j in range(len(f1_kfold)):
			print("\nFold "+str(j+1)+": ")
			print("F1: ", f1_kfold[j])
			print("Precision: ", precision_kfold[j])
			print("Recall: ", recall_kfold[j])		
			print("Thresh: ", thresh_kfold[j])
			print("Acc: ", acc_kfold[j])
			print("Acc Thresh: ", thresh_acc_kfold[j])
			if filtering == "Y":
				print("\nFiltered F1: ", f1_filter_kfold[j])
				print("Filtered Precision: ", precision_filter_kfold[j])
				print("Filtered Recall: ", recall_filter_kfold[j])
				print("Filtered Thresh: ", thresh_filter_kfold[j])
				print("Filtered Acc: ", acc_filter_kfold[j])
				print("Filtered Acc Thresh: ", thresh_acc_filter_kfold[j])
				print("\nCase 0 Predictions: ", error_types_kfold[j][0])
				print("Case 1 Predictions: ", error_types_kfold[j][1])
				print("Case 2 or 3 Predictions: ", error_types_kfold[j][2])			

		print("\n-----Settings-----")
		print("Root: ", root)
		print("Classifier: ", algo)
		print("Sample Threshold: ", sample_threshold)
		print("K-mer size: ", kmer)
		print("Dataset: ", dataset)
		print("Filtering: ", filtering)
		print("\nTotal time: ", time()-time_start_all)		
		print("Script: ", sys.argv[0])
		print("Taxonomy data: ", f1)
		print("Sequence data: ", f2)
		print("Folds: ", folds)
		print("\nDONE!\n")
		
		f = open("log_kfold_"+root+"_"+algo+"_"+str(kmer)+"_"+dataset,"w")
		print("Saving performance stats to file...")
		print("\n-----Results-----", file=f)
		print("Avg F1: ", np.mean(np.array(f1_kfold)), file=f)
		print("Avg Precision: ", np.mean(np.array(precision_kfold)), file=f)
		print("Avg Recall: ", np.mean(np.array(recall_kfold)), file=f)
		print("Avg Threshold: ", np.mean(np.array(thresh_kfold)), file=f)
		print("Avg Accuracy: ", np.mean(np.array(acc_kfold)),file=f)
		print("Avg Accuracy Thresh: ", np.mean(np.array(thresh_acc_kfold)), file=f)
		
		if filtering == "Y":
			print("\nAvg Filtered F1: ", np.mean(np.array(f1_filter_kfold)), file=f)
			print("Avg Filtered Precision: ", np.mean(np.array(precision_filter_kfold)), file=f)
			print("Avg Filtered Recall: ", np.mean(np.array(recall_filter_kfold)), file=f)
			print("Avg Filtered Threshold: ", np.mean(np.array(thresh_filter_kfold)), file=f)
			print("Avg Filtered Accuracy: ", np.mean(np.array(acc_filter_kfold)), file=f)
			print("Avg Filtered Accuracy Thresh: ", np.mean(np.array(thresh_acc_filter_kfold)), file=f)
			means = np.mean(np.array(error_types_kfold), axis=0)
			print("\nAvg Case 0 Predictions: ", means[0], file=f)
			print("Avg Case 1 Predictions: ", means[1], file=f)
			print("Avg Case 2 or 3 Predictions: ", means[2], file=f)

		for j in range(len(f1_kfold)):
			print("\nFold "+str(j+1)+": ", file=f)
			print("Train size: ", train_size_kfold[j], file=f)
			print("Test size: ", test_size_kfold[j], file=f)
			print("F1: ", f1_kfold[j], file=f)
			print("Precision: ", precision_kfold[j],file=f)
			print("Recall: ", recall_kfold[j],file=f)      
			print("Thresh: ", thresh_kfold[j],file=f)
			print("Acc: ", acc_kfold[j], file=f)
			print("Acc Thresh: ", thresh_acc_kfold[j], file=f)
				
			if filtering == "Y":                    
				print("\nFiltered F1: ", f1_filter_kfold[j], file=f)
				print("Filtered Precision: ", precision_kfold[j], file=f)
				print("Filtered Recall: ", recall_kfold[j], file=f)
				print("Filtered Thresh: ", thresh_kfold[j], file=f)
				print("Filtered Acc: ", acc_filter_kfold[j], file=f)
				print("Filtered Acc Thresh: ", thresh_acc_filter_kfold[j], file=f)
				print("\nCase 0 Predictions: ", error_types_kfold[j][0], file=f)
				print("Case 1 Predictions: ", error_types_kfold[j][1], file=f)
				print("Case 2 or 3 Predictions: ", error_types_kfold[j][2], file=f)

		print("\n-----Settings-----", file=f)
		print("Root: ", root, file=f)
		print("Classifier: ", algo, file=f)
		print("Sample Threshold: ", sample_threshold, file=f)
		print("K-mer size: ", kmer, file=f)
		print("Dataset: ", dataset, file=f)
		print("Filtering: ", filtering, file=f)
		print("Folds: ", folds, file=f)
		print("\nTotal time: ", time()-time_start_all, file=f)
		print("Script: ", sys.argv[0], file=f)
		print("Taxonomy data: ", f1, file=f)
		print("Sequence data: ", f2, file=f)
		print("\nDONE!\n")




