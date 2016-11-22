from __future__ import division
from __future__ import print_function
import json
import networkx as nx
import collections
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

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


def convert_ontology_to_graph(root):
	graph = nx.Graph()
	descendants = go_descendants[root]
	for node in go_ontology:
		node_id = node["id"]
		if node_id in descendants:
			children = go_children[node_id]
			graph.add_node(node_id)
			for c in children:
				graph.add_edge(node_id, c)
	return graph

def convert_ontology_to_digraph(root):
	graph = nx.DiGraph()
	descendants = go_descendants[root]
	for node in go_ontology:
		node_id = node["id"]
		if node_id in descendants:
			children = go_children[node_id]
			graph.add_node(node_id)
			for c in children:
				graph.add_edge(node_id, c)
	return graph

#get distance to root
def get_distance_from_root(root):
	ontology_distance = {}
	q = collections.deque()
	q.append(root)
	nodes_seen = list()
	ontology_distance[root] = 0
	while len(q)>0:
		parent = q.popleft()
		children = go_children[parent]
		for child in children:
			if child not in nodes_seen:
				if parent == root:
					ontology_distance[child] = 1
				else:
					ontology_distance[child] = ontology_distance[parent]+1
				nodes_seen.append(child)
				q.append(child)
	return ontology_distance

#####################################################################

cc_term = "GO:0005575"
mf_term = "GO:0003674"
bp_term = "GO:0008150"

graph = convert_ontology_to_graph(cc_term)
digraph = convert_ontology_to_digraph(cc_term)

cc_degree_hist = nx.degree_histogram(graph)

f = open("results/predicted_labels_kfold_GO_C_U_N_0.07.json")
pred_labels = json.load(f)
f = open("results/true_labels_kfold_GO_C_U_N_full.json")
true_labels = json.load(f)
tkeys = true_labels.keys()
pkeys = pred_labels.keys()
results_dict = {}
for i in pkeys:
	predicted = pred_labels[i]
	for pred in predicted:
		score = 0
		if pred in true_labels[i]:
			score = 1
		if pred not in results_dict.keys():
			results_dict[pred] = {}
			results_dict[pred]["correct"] = score
			results_dict[pred]["predicted"] = 1
		else:
			results_dict[pred]["correct"]+=score
			results_dict[pred]["predicted"]+=1
for i in tkeys:
	tr = true_labels[i]
	for t in tr:
		if t not in results_dict.keys():
			results_dict[t]["true"] = 0
		results_dict[t]["true"]+=1
for k in results_dict.keys():
	results_dict[k]["precision"] = results_dict[k]["correct"]/results_dict[k]["predicted"]
	results_dict[k]["recall"] = results_dict[k]["correct"]/results_dict[k]["true"]
	p = results_dict[k]["precision"]
	r = results_dict[k]["recall"]
	results_dict[k]["f1"] = (2*p*r/(p+r))

distances = get_distance_from_root(cc_term)
results_keys = results_dict.keys()

#### Plot precision vs. degree centrality ####
degree_centrality = nx.degree_centrality(graph)
xdata = []
ydata = []
dist = []

for k in results_keys:
	dist.append(distances[k])
	xdata.append(results_dict[k]["precision"])
	if k in degree_centrality.keys():
		ydata.append(degree_centrality[k])
fig = plt.figure()
ax = fig.add_subplot(111)
plt.scatter(xdata, ydata, c=dist)
plt.title("Cellular Component - Uniprot data")
plt.xlabel("Precision per node")
plt.ylabel("Degree centrality per node")
fig.savefig("cc_degree_centrality.png")

#### Plot precision vs. closeness centrality ####
close_centrality = nx.closeness_centrality(graph)
xdata = []
ydata = []
dist = []
for k in results_keys:
	dist.append(distances[k])
	xdata.append(results_dict[k]["precision"])
	if k in close_centrality.keys():
		ydata.append(close_centrality[k])
fig = plt.figure()
ax = fig.add_subplot(111)
#ax.scatter(xdata, ydata)
plt.scatter(xdata, ydata, c=dist)
plt.title("Cellular Component - Uniprot data")
plt.xlabel("Precision per node")
plt.ylabel("Closeness centrality per node")
fig.savefig("cc_close_centrality.png")

#### Plot precision vs. betweenness centrality ####
between_centrality = nx.betweenness_centrality(graph)
xdata = []
ydata = []
dist = []
for k in results_keys:
	dist.append(distances[k])
	xdata.append(results_dict[k]["precision"])
	if k in between_centrality.keys():
		ydata.append(between_centrality[k])
fig = plt.figure()
ax = fig.add_subplot(111)
plt.scatter(xdata, ydata, c=dist)
plt.title("Cellular Component - Uniprot data")
plt.xlabel("Precision per node")
plt.ylabel("Betweenness centrality per node")
fig.savefig("cc_between_centrality.png")

#### Plot precision vs. load centrality ####
load_centrality = nx.load_centrality(graph)
xdata = []
ydata = []
dist = []
for k in results_keys:
	dist.append(distances[k])
	xdata.append(results_dict[k]["precision"])
	if k in load_centrality.keys():
		ydata.append(load_centrality[k])
fig = plt.figure()
ax = fig.add_subplot(111)
plt.scatter(xdata, ydata, c=dist)
plt.title("Cellular Component - Uniprot data")
plt.xlabel("Precision per node")
plt.ylabel("Load centrality per node")
fig.savefig("cc_load_centrality.png")


#############################################################################

mf_graph = convert_ontology_to_graph(mf_term)
mf_digraph = convert_ontology_to_digraph(mf_term)

mf_degree_hist = nx.degree_histogram(mf_graph)

f = open("results/predicted_labels_kfold_GO_F_U_N_0.01.json")
pred_labels = json.load(f)
f = open("results/true_labels_kfold_GO_F_U_full.json")
true_labels = json.load(f)

tkeys = true_labels.keys()
pkeys = pred_labels.keys()
mf_results_dict = {}
for i in pkeys:
	predicted = pred_labels[i]
	tr = true_labels[i]
	for pred in predicted:
		score = 0
		if pred in true_labels[i]:
			score = 1
		if pred not in mf_results_dict.keys():
			mf_results_dict[pred] = {}
			mf_results_dict[pred]["correct"] = score
			mf_results_dict[pred]["predicted"] = 1
			mf_results_dict[pred]["true"] = 0
		else:
			mf_results_dict[pred]["correct"]+=score
			mf_results_dict[pred]["predicted"]+=1
for i in tkeys:
	tr = true_labels[i]
	for t in tr:
		if t in mf_results_dict.keys():
			mf_results_dict[t]["true"]+=1


for k in mf_results_dict.keys():
	mf_results_dict[k]["precision"] = mf_results_dict[k]["correct"]/mf_results_dict[k]["predicted"]
	mf_results_dict[k]["recall"] = mf_results_dict[k]["correct"]/mf_results_dict[k]["true"]
	p = mf_results_dict[k]["precision"]
	r = mf_results_dict[k]["recall"]
	if p+r>0:
		mf_results_dict[k]["f1"] = (2*p*r/(p+r))
	else:
		mf_results_dict[k]["f1"] = 0

mf_distances = get_distance_from_root(mf_term)
mf_results_keys = mf_results_dict.keys()


#### Plot precision vs. degree centrality ####
mf_degree_centrality = nx.degree_centrality(mf_graph)
xdata = []
ydata = []
dist = []

for k in mf_results_keys:
	if k in mf_distances.keys():
		dist.append(mf_distances[k])
		xdata.append(mf_results_dict[k]["precision"])
		if k in mf_degree_centrality.keys():
			ydata.append(mf_degree_centrality[k])

fig = plt.figure()
ax = fig.add_subplot(111)
plt.scatter(xdata, ydata, c=dist)
plt.title("Molecular Function - Uniprot data")
plt.xlabel("Precision per node")
plt.ylabel("Degree centrality per node")
fig.savefig("mf_degree_centrality.png")

#### Plot precision vs. closeness centrality ####
mf_close_centrality = nx.closeness_centrality(mf_graph)
xdata = []
ydata = []
dist = []

for k in mf_results_keys:
	if k in mf_distances.keys():
		dist.append(mf_distances[k])
		xdata.append(mf_results_dict[k]["precision"])
		if k in mf_close_centrality.keys():
			ydata.append(mf_close_centrality[k])

fig = plt.figure()
ax = fig.add_subplot(111)
#ax.scatter(xdata, ydata)
plt.scatter(xdata, ydata, c=dist)
plt.title("Molecular Function - Uniprot data")
plt.xlabel("Precision per node")
plt.ylabel("Closeness centrality per node")
fig.savefig("mf_close_centrality.png")

#### Plot precision vs. betweenness centrality ####
mf_between_centrality = nx.betweenness_centrality(mf_graph)
xdata = []
ydata = []
dist = []

for k in mf_results_keys:
	if k in mf_distances.keys():
		dist.append(mf_distances[k])
		xdata.append(mf_results_dict[k]["precision"])
		if k in mf_between_centrality.keys():
			ydata.append(mf_between_centrality[k])

fig = plt.figure()
ax = fig.add_subplot(111)
plt.scatter(xdata, ydata, c=dist)
plt.title("Molecular Function - Uniprot data")
plt.xlabel("Precision per node")
plt.ylabel("Betweenness centrality per node")
fig.savefig("mf_between_centrality.png")

#### Plot precision vs. load centrality ####
mf_load_centrality = nx.load_centrality(mf_graph)
xdata = []
ydata = []
dist = []

for k in mf_results_keys:
	if k in mf_distances.keys():
		dist.append(mf_distances[k])
		xdata.append(mf_results_dict[k]["precision"])
		if k in mf_load_centrality.keys():
			ydata.append(mf_load_centrality[k])

fig = plt.figure()
ax = fig.add_subplot(111)
plt.scatter(xdata, ydata, c=dist)
plt.title("Molecular Function - Uniprot data")
plt.xlabel("Precision per node")
plt.ylabel("Load centrality per node")
fig.savefig("mf_load_centrality.png")


######################################################################

bp_graph = convert_ontology_to_graph(bp_term)
bp_digraph = convert_ontology_to_digraph(bp_term)




######################################################################

# plot degree histograms for each ontology

cc_degree_hist = nx.degree_histogram(graph)
fig = plt.figure()
ax = fig.add_subplot(111)
plt.bar(range(100), cc_degree_hist[:100])
plt.xlabel("Degree")
plt.ylabel("Degree frequency")
plt.title("Cellular Component - Degree Histogram")
fig.savefig("cc_degree_hist.png")


mf_degree_hist = nx.degree_histogram(mf_graph)
fig = plt.figure()
ax = fig.add_subplot(111)
plt.bar(range(100), mf_degree_hist[:100])
plt.xlabel("Degree")
plt.ylabel("Degree frequency")
plt.title("Molecular Function - Degree Histogram")
fig.savefig("mf_degree_hist.png")


bp_degree_hist = nx.degree_histogram(bp_graph)
fig = plt.figure()
ax = fig.add_subplot(111)
plt.bar(range(100), bp_degree_hist[:100])
plt.xlabel("Degree")
plt.ylabel("Degree frequency")
plt.title("Biological Process - Degree Histogram")
fig.savefig("bp_degree_hist.png")


##############################################################


#check for under-predictions
distances = get_distance_from_root(cc_term)
under_pred = []
for k in pkeys:
	if len(pred_labels[k]) < len(true_labels[k]):
		#we under-predicted: look for the most specific prediction
		d = [distances[term] for term in true_labels[k]]
		lowest_label = true_labels[k][d.index(max(d))]
		parents_lowest_label = go_parents[lowest_label]
		if (len(parents_lowest_label))>1:
			inter = set(pred_labels[k]) & set(parents_lowest_label)
			if len(inter)>1:
				under_pred.append(k)
				print("\nKey: ", k)
				print("Pred: ", pred_labels[k])
				print("True: ", true_labels[k])

