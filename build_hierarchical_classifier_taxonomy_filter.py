from __future__ import division
from __future__ import print_function

import numpy as np
import collections
import json
import string
import re
import sys, getopt
import pickle
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
		pred_leaf = pred_labels[pred_path_len.index(max_path)]
		if true_leaf in pred_labels:
			acc = 1
	else:
		true_leaf = true_labels[true_path_len.index(max(true_path_len))]
		pred_leaf=pred_labels[pred_path_len.index(max(pred_path_len))]
		acc = -1
	return acc,true_leaf,pred_leaf


#compute precision & recall for one test point
def evaluate_prediction(true_labels, pred_labels):
	tp = set(pred_labels).intersection(true_labels)
	precision = 0.0
	recall = 0.0
	if len(pred_labels)!=0:
		precision = len(tp)/len(pred_labels)
	if len(true_labels)!=0:
		recall = len(tp)/len(true_labels)
	return precision,recall


# propagate the given label/rank upwards in the taxonomy until it reaches a root
def propagate_labels(labels,taxonomy):
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
			positive_prob = prob[classes[:]==1][0]
		else:
			if classes[0]==0:
				positive_prob = 0.0
			else:
				positive_prob = 1.0
		prob_ontology[key] = positive_prob
	return prob_ontology



######################################  MAIN  ############################################


if __name__ == "__main__":
	if len(sys.argv[1:]) < 6:
		print("This script requires 5 arguments: root, kmer size, sample_threshold, classifier, dataset, filter")
	else:
		time_start_all = time()
		print("\n=====START=====\n")
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
		else:
			print("Dataset: SILVA")
		
		filtering = sys.argv[6]
		print("Filter: ", filtering)
		
		#from this point, all nodes above the root will not be considered
		if dataset == "R":
			f1 = open("rdp_taxonomy.json","r")
			f2 = open("rdp_data_dict.json", "r")
		else:
			f1 = open("taxonomy.json", "r")
			f2 = open("silva_dict.json", "r")
		

		taxonomy = json.load(f1)
		data_dict = json.load(f2)

		#Only use the parts of the dataset that belongs to the root
		desc_root = taxonomy[root]["descendants"]
		print("Descendants: ", len(desc_root))

                sequence_ids = list()
                sequence = list()
                classification = list()
		
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

		#divide dataset
		index = int(len(sequence_ids)/5)*4

		X_train = sequence[:index]
		sequence_ids_train = sequence_ids[:index]
		classification_train = classification[:index]

		X_test_temp = sequence[index:]
		sequence_ids_test_temp = sequence_ids[index:]
		classification_test_temp = classification[index:]
		
		X_test = []
		sequence_ids_test = []
		classification_test = []
		for i in range(len(X_test_temp)):
			if classification_train.count(classification_test_temp[i])>=2:
				X_test.append(X_test_temp[i])
				sequence_ids_test.append(sequence_ids_test_temp[i])
				classification_test.append(classification_test_temp[i])

		print("Train set: ", len(X_train))
		print("Test set: ", len(X_test))
	
		#vectorize features
		vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5)
		X_train = vectorizer.fit_transform(X_train)
		X_test = vectorizer.transform(X_test)

		#create binary classifiers
		print("Creating classifiers")
		time_start_classifier = time()
		classifiers = {}
		positive_count = {}
		true_species = []
		pred_species = []
		for desc in desc_root:
			descendants = taxonomy[desc]["descendants"]
			y_train = list()
			for c in classification_train:
				if c in descendants:
					y_train.append(1)
				else:
					y_train.append(0)
			pos_count = y_train.count(1)
			if pos_count>=sample_threshold:
				if algo != "S" or pos_count==len(y_train):
					clf = MultinomialNB(alpha=.01).fit(X_train, y_train)
					classifiers[desc] = clf
				else:
					clf = svm.SVC(probability=True)
					clf.fit(X_train,y_train)
					classifiers[desc] = clf
				positive_count[desc] = pos_count
		print("Done creating classifiers. No. of classifiers: ", len(classifiers))
		time_end_classifier = time()-time_start_classifier

		#run classifiers on the test set
		print("Running classifiers on the test set")
		time_start_test = time()
		prob_dict = {}
		for i in range(len(sequence_ids_test)):
			test_point = X_test[i]
			prob_dict[sequence_ids_test[i]] = predict_class(test_point)
		time_end_test = time()-time_start_test
		
		#print("Saving prob_dict to file...")
		#fname = "results/prob_dict_"+root+"_"+str(kmer)+"_"+str(sample_threshold)+"_"+algo+"_"+dataset+".json"
		#with open(fname, "w") as f:
		#	json.dump(prob_dict,f)	
		#	print("OK!")	
	
		#plot the precision/recall/f1 vs threshold
		print("Calculating F1/recall/precision by threshold")
		time_start_eval = time()
		results_dict = {}
		precision_list = []
		recall_list = []
		acc_list = []
		f1_list = []
		pred_species = []
		true_species = []
		if filtering == "Y":
			results_filter_dict = {}
			precision_filter_list = []
			recall_filter_list = []	
			acc_filter_list = []	
			f1_filter_list = []			
			error_types = []			

		for thresh in range(0, 101):
			thresh = float(thresh)/100
			print("thresh: ", thresh)
			total_precision = 0
			total_recall = 0
			total_acc = 0
			if filtering == "Y":
				total_precision_filter = 0
				total_recall_filter = 0
                        	total_acc_filter = 0
				classified_total_filter = 0
				classified_total = 0
				type_0 = 0
                        	type_1 = 0
			for i in range(len(sequence_ids_test)):
				true_labels = propagate_labels([classification_test[i]],taxonomy)
				prob_ontology = prob_dict[sequence_ids_test[i]]
				positive_labels = list()
				for key in classifiers.keys():
					if prob_ontology[key] >= thresh:
						positive_labels.append(key)
				if filtering == "Y":
					filtered_labels,error_type = filter_prediction(positive_labels,i, taxonomy)
					if error_type == 0:
						type_0+=1
					elif error_type == 1:
						type_1+=1
					results_filter_dict[sequence_ids_test[i]] = {}
					results_filter_dict[sequence_ids_test[i]]["true"]=true_labels
                                	results_filter_dict[sequence_ids_test[i]]["predicted"]=filtered_labels
					precision_filter,recall_filter = evaluate_prediction(true_labels, filtered_labels)
                                	acc_filter,true_sp,pred_sp = compute_accuracy(true_labels, filtered_labels, taxonomy)
					total_precision_filter+=precision_filter
                                	total_recall_filter+=recall_filter
					if acc_filter >= 0:
						classified_total_filter+=1
						total_acc_filter+=acc_filter				
					
				predicted_labels = propagate_labels(positive_labels,taxonomy)
				results_dict[sequence_ids_test[i]] = {}
				results_dict[sequence_ids_test[i]]["true"]=true_labels
				results_dict[sequence_ids_test[i]]["predicted"]=predicted_labels
				precision,recall = evaluate_prediction(true_labels, predicted_labels)
				acc,true_sp,pred_sp = compute_accuracy(true_labels, predicted_labels, taxonomy)
				total_precision+=precision
				total_recall+=recall
				if acc >= 0:
					true_species.append(true_sp)
					pred_species.append(pred_sp)
					#print("True species: ", true_sp)
					#print("Pred species: ", pred_sp)
					classified_total+=1
					total_acc+=acc
		
			#with open("results/results_no_filter_"+root+"_"+str(kmer)+"_"+str(thresh)+".json","w") as f:
			#	json.dump(results_dict,f)
			#with open("results/results_no_filter_true_species_"+root+"_"+str(kmer)+"_"+str(thresh)+".json","w") as f:			
			#	json.dump(true_species,f)
			#with open("results/results_no_filter_pred_species_"+root+"_"+str(kmer)+"_"+str(thresh)+".json","w") as f:				
			#	json.dump(pred_species,f)		

			if filtering == "Y":
                        	final_precision_filter = total_precision_filter/len(sequence_ids_test)
                        	final_recall_filter = total_recall_filter/len(sequence_ids_test)
                        	final_acc_filter = total_acc_filter/classified_total_filter
				final_f1_filter = (2*final_precision_filter*final_recall_filter)/(final_precision_filter+final_recall_filter)
                        	precision_filter_list.append(final_precision_filter)
                        	recall_filter_list.append(final_recall_filter)
				acc_filter_list.append(final_acc_filter)
                        	f1_filter_list.append(final_f1_filter)							
				error_types.append([type_0, type_1])
			
			final_precision = total_precision/len(sequence_ids_test)
			final_recall = total_recall/len(sequence_ids_test)
			final_acc = total_acc/classified_total
			final_f1 = (2*final_precision*final_recall)/(final_precision+final_recall)
			precision_list.append(final_precision)
			recall_list.append(final_recall)
			acc_list.append(final_acc)
			f1_list.append(final_f1)
			print("Classified total: ", classified_total)
			print("Total accuracy: ", total_acc)
			print("Final accuracy: ", final_acc)			

		if filtering == "Y":
                	max_f1_filter = max(f1_filter_list)
                	max_thresh_filter = f1_filter_list.index(max_f1_filter)
                	max_precision_filter = precision_filter_list[max_thresh_filter]
                	max_recall_filter = recall_filter_list[max_thresh_filter]			
			max_acc_filter = acc_filter_list[max_thresh_filter]
			max_type_0 = error_types[max_thresh_filter][0]
			max_type_1 = error_types[max_thresh_filter][1]
			max_type_n = len(sequence_ids_test)-(max_type_1+max_type_0)			
			best_acc_filter =max(acc_filter_list)
			best_acc_filter_thresh = acc_filter_list.index(best_acc_filter)			

		max_f1 = max(f1_list)
		max_thresh = f1_list.index(max_f1)
		max_precision = precision_list[max_thresh]
		max_recall = recall_list[max_thresh]
		max_acc = acc_list[max_thresh]
		best_acc = max(acc_list)
		best_acc_thresh = acc_list.index(best_acc)
		time_end_eval = time()-time_start_eval
		time_end_all = time()-time_start_all
			

		print("\n-----Results-----")
		print("Max F1: ", max_f1)
		print("Max Precision: ", max_precision)
		print("Max Recall: ", max_recall)
		print("Max Accuracy: ", max_acc)
		print("Max Threshold: ", max_thresh)
		print("Best Accuracy: ", best_acc)
		print("Best Accuracy Thresh: ", best_acc_thresh)		

		if filtering == "Y":
                	print("\nFiltered Max F1: ", max_f1_filter)
                	print("Filtered Max Precision: ", max_precision_filter)
                	print("Filtered Max Recall: ", max_recall_filter)
                	print("Filtered Max Accuracy: ", max_acc_filter)
			print("Filtered Max Threshold: ", max_thresh_filter)		
			print("Filtered Best Accuracy: ", best_acc_filter)
			print("Filtered Best Accuracy Thresh: ", best_acc_filter_thresh)			

			print("\nType 0 predictions: ", max_type_0)
			print("Type 1 predictions: ", max_type_1)			
			print("Type 2 or 3 predictions: ", max_type_n)			

		print("\n-----Timings-----")
		print("Creating classifiers: ", time_end_classifier)
		print("Evaluating test set: ", time_end_test)
		print("Computing metrics per threshold: ", time_end_eval)
		print("Total time: ", time_end_all)
		
		print("\n-----Settings-----")
		print("Root: ", root)
		print("K-mer size: ", kmer)
		print("Sample Threshold: ", sample_threshold)
		if algo == "S":
			print("Classifier Type: SVM")
		else:
			print("Classifier Type: Naive-Bayes")
		if dataset == "R":
			print("Dataset: RDP")
		else:
			print("Dataset: SILVA")
		print("Filtering: ", filtering)
		print("Classifiers: ", len(classifiers))
		print("Script: ", sys.argv[0])
                print("Taxonomy data: ", f1)
                print("Sequence data: ", f2)
		print("Saving results_dict to file...")
		with open("results/results_no_filter"+root+"_"+str(kmer)+"_"+str(sample_threshold)+"_"+algo+"_"+dataset+".json","w") as f:
			json.dump(results_dict, f)
			print("OK!")
		if filtering =="Y":
			print("Saving results_filter_dict to file...")
			with open("results/results_filter_"+root+"_"+str(kmer)+"_"+str(sample_threshold)+"_"+algo+"_"+dataset+".json","w") as f:
                        	json.dump(results_filter_dict, f)
                        	print("OK!")
		#print("Saving vectorizer to file...")
		#with open("classifiers/vectorizer_"+root+"_"+str(kmer)+"_"+str(sample_threshold)+"_"+algo+"_"+dataset,"w") as f:
		#	pickle.dump(vectorizer, f)
		#	print("OK!")
		#print("Saving classifiers to file...")
		#keys = classifiers.keys()
		#for key in keys:
		#	with open("classifiers/classifiers_"+key+"_"+root+"_"+str(kmer)+"_"+str(sample_threshold)+"_"+algo+"_"+dataset,"w") as f:
		#		pickle.dump(classifiers[key], f)
		#print("OK!")

		f = open("log_"+str(time()), "w")
		print("Saving performance stats...")
		print("\n-----Results-----", file=f)
		print("Max F1: ", max_f1, file=f)
		print("Max Precision: ", max_precision,file=f)
		print("Max Recall: ", max_recall, file=f)
                print("Max Accuracy: ", max_acc, file=f)
		print("Max Threshold: ", max_thresh, file=f)

                if filtering == "Y":
                        print("\nFiltered Max F1: ", max_f1_filter, file=f)
                        print("Filtered Max Precision: ", max_precision_filter,file=f)
                        print("Filtered Max Recall: ", max_recall_filter,file=f)
                	print("Filtered Max Accuracy: ", max_acc_filter, file=f)
                        print("Filtered Max Threshold: ", max_thresh_filter,file=f)

                        print("\nType 0 predictions: ", max_type_0, file=f)
                        print("Type 1 predictions: ", max_type_1, file=f)  
                        print("Type 2 or 3 predictions: ", max_type_n, file=f)

		print("\n-----Timings-----", file=f)
		print("Creating classifiers: ", time_end_classifier, file=f)
		print("Evaluating test set: ", time_end_test, file=f)
		print("Computing metrics per threshold: ", time_end_eval, file=f)
		print("Total time: ", time_end_all, file=f)

		print("\n-----Settings-----",file=f)
		print("Root: ", root, file=f)
		print("K-mer size: ", kmer, file=f)
		print("Sample Threshold: ", sample_threshold, file=f)
		if algo == "S":
				print("Classifier Type: SVM", file=f)
		else:
				print("Classifier Type: Naive-Bayes", file=f)
		if dataset == "R":
			print("Dataset: RDP", file=f)
		else:
			print("Dataset: SILVA", file=f)
		print("Filtering: ", filtering, file=f)
		print("Classifiers: ", len(classifiers),file=f)
		print("Script: ", sys.argv[0],file=f)
		print("Taxonomy data: ", f1, file=f)
		print("Sequence data: ", f2, file=f)
		print("DONE!\n")


