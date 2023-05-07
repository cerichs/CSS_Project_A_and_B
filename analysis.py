import networkx as nx
import pickle
import random
import community
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import colorsys
from netwulf import visualize
import pandas as pd
from sklearn.metrics import confusion_matrix

def clean_family(string):
    if string is None:
        return None
    string = string.replace("†","").replace(" (","")
    count = 0
    for i,chara in enumerate(string):
        if chara.isupper():
            count += 1
            if count > 1:
                return string[:i]
    return string

def add_attr(Graph,attr_dict):
    to_remove =[]
    for Names in Graph.nodes:
        if Names in attr_dict:
            Graph.nodes[Names]['Class'], Graph.nodes[Names]['Order'], Graph.nodes[Names]['Superfamily'], Graph.nodes[Names]['Family'], _ = attr_dict[Names].values()
            Graph.nodes[Names]['Class'] = clean_family(Graph.nodes[Names]['Class']) if Graph.nodes[Names]['Class'] is not None else None
        else:
            to_remove.append(Names) # Some nodes get added to graph even though they are redirects, the cause is known but no good way to handle it
    for names in to_remove:
        Graph.remove_node(names)
    return Graph

def same_class(G):
    same_class_fractions = []
    for node in G.nodes():
        same_class_neighbors = 0
        total_neighbors = 0
        
        for neighbor in G.neighbors(node):
            if G.nodes[neighbor]["Class"] == G.nodes[node]["Class"]:
                same_class_neighbors += 1
            total_neighbors += 1
        
        if total_neighbors > 0:
            same_field_fraction = same_class_neighbors / total_neighbors
        else:
            same_field_fraction = 0
        same_class_fractions.append(same_field_fraction)
    return same_class_fractions

def same_class_rand(Graph):
    shuffled_G = Graph.copy()
    Family = [Graph.nodes[node]["Class"] for node in Graph.nodes()]
    random.shuffle(Family)

    for i, node in enumerate(shuffled_G.nodes()):
        shuffled_G.nodes[node]["Class"] = Family[i]
    return shuffled_G

def same_class_rand_n(Graph, n = 250):
    results = []
    Family = [Graph.nodes[node]["Class"] for node in Graph.nodes()]
    for i in range(n):
        temp = same_class_rand(Graph)
        results.append(np.mean(same_class(temp)))
    plt.hist(results,label = "Random", bins = 20)
    plt.xlabel("Fraction")
    plt.ylabel("Frequency")
    plt.title("Fraction of edges in same Class")
    plt.show()

def compute_modularity(Graph,partitioning):
    L = Graph.number_of_edges() * 2 # In = out
    M = 0
    communities = set(partitioning.values())

    for community in communities:
        nodes_in_community = [node for node, community_id in partitioning.items() if community_id == community]
        subgraph = Graph.subgraph(nodes_in_community)
        k_c = sum(dict(subgraph.degree()).values())
        L_c = subgraph.number_of_edges()
        M += L_c / L - (k_c / (2 * L)) ** 2

    return M

def double_edge_swap(GG,N):
    i = 0
    G_copy = GG.copy()
    while(i<N):
        edges = list(G_copy.edges()) # update edges after adding and removal
        (u, v), (x, y) = random.sample(edges, 2) # picking two random edges
        if (u != v) and (v != x) and (u, y) not in G_copy.edges() and (x, v) not in G_copy.edges(): # checking conditions
            G_copy.add_edge(u, y)
            G_copy.add_edge(x, v)
            G_copy.remove_edge(u, v)
            G_copy.remove_edge(x, y)
        i+=1
        if i % 1000 == 0:
            print(i)
    return G_copy

def assortative_matrix(Graph,unique_labels):
    matrix = np.zeros((len(unique_labels),len(unique_labels)))

    values = nx.get_node_attributes(Graph, "Class").values()
    num_values = len(values)

    for start, end in Graph.edges(): # Looping over all edges in graph (since its undirected its not really start and stop)
        x = Graph.nodes[start]["Class"] # Getting the start point of the edge 
        y = Graph.nodes[end]["Class"] # Getting the end point of the edge 
        if x in unique_labels: 
            x = unique_labels[x]
        else:
            x = unique_labels[None] # in case x is nan
            
        if y in unique_labels:
            y = unique_labels[y]
        else:
            y = unique_labels[None] # in case y is nan
        matrix[x, y] += 1 
        
    num_edges = len(Graph.edges())
    matrix /= num_edges # averaging the occurence with the total edges

    trace = np.trace(matrix) # trace of the matrix, the sum of the diagonal entries
    mix_matrix = np.sum(np.matmul(matrix, matrix))

    r1 = (trace-mix_matrix)/(1-mix_matrix) # Eq. 2

    return r1

if __name__ == "__main__":
    # Load Network
    with open('data/all_animal_to_all_animal.pickle', 'rb') as handle:
        b = pickle.load(handle)
    with open('data/animal_attributes.pickle', 'rb') as handle:
        c = pickle.load(handle)
    banned_set = {"Animal","Reptile","Arthropod","Chordate","Bird","Rodent","Insect"}
    for items in c:
        banned_set.add(clean_family(c[items]["Class:"]))
        banned_set.add(clean_family(c[items]["Order:"]))
        banned_set.add(clean_family(c[items]["Superfamily:"]))
        banned_set.add(clean_family(c[items]["Family:"]))
    #edgelist = [None]*len(b)
    banned_set = {}
    edgelist = []
    for i,items in enumerate(b):
        if items[0] not in banned_set and items[1] not in banned_set:
            edgelist.append([items[0].replace("/wiki/", ""),items[1].replace("/wiki/", ""),int(b[items])])
    G_reptile = nx.Graph()
    

    G_reptile.add_weighted_edges_from(edgelist)
    print(G_reptile)
    G_reptile_attr = add_attr(G_reptile,c)

    # Same class fractions
    same_class_fractions = same_class(G_reptile_attr)
    print(f"The average fraction is: {np.mean(same_class_fractions):.3f}")

    # Same class fractions for random graph
    same_class_fractions_rand = same_class_rand(G_reptile_attr)
    print(f"The average fraction is: {np.mean(same_class(same_class_fractions_rand)):.3f}")

    # Same class fractions
    #same_family_rand_n(G_reptile_attr, n = 250)

    # Assortativity coefficient 
    labels = np.unique([G_reptile_attr.nodes[node]["Class"] for node in G_reptile_attr.nodes() if G_reptile_attr.nodes[node]["Class"] is not None])
    unique_labels = {name: i for i,name in enumerate(labels)}
    unique_labels[None] = len(unique_labels)
    r1 = assortative_matrix(G_reptile_attr,unique_labels)
    print(f"Assortativity coefficient: {r1:.3f}")

    # Modularity Family split of Reptile Graph
    Family_split = nx.get_node_attributes(G_reptile_attr, "Family")

    #Then we compute the modularity of the partitioning by using the function above
    modularity = compute_modularity(G_reptile_attr, Family_split)

    print(f"The modularity of the family split partitioning is {modularity:.3f}")
    """
    G_new = double_edge_swap(G_reptile,len(list(G_reptile.edges())))
    degree_1 = []
    for node in G_reptile.nodes():
        degree_1.append(G_reptile.degree(node))
    degree_2 = []
    for node in G_new.nodes():
        degree_2.append(G_new.degree(node))

    print(degree_1 == degree_2)
    modularity = compute_modularity(G_new, Family_split)
    print(f"The modularity of the randomly shuffled network: {modularity:.3f}")
    """
    #Louvain algorithm
    partition = community.best_partition(G_reptile,random_state=42) # dictionary, keys are the nodes and values are communities for each node

    # modularity of partition
    modularity = community.modularity(partition, G_reptile) 

    size = []
    for community_ in set(partition.values()):
        temp = []
        for part in partition:
            if partition[part] == community_:
                temp.append(part)
        size.append(len(temp))

    print("Number of communities:", len(set(partition.values())))
    print("Community sizes:", sorted(size,reverse=True))
    print(f"Modularity: {modularity:.2f}")

    louvain_labels = list(partition.values())

    #ground truth community labels as a list
    gt_labels = [G_reptile_attr.nodes[node_id]["Class"] for node_id in G_reptile_attr.nodes()]
    gt_labels = [labels if labels is not None else "None" for labels in gt_labels] # convert None to string(None)

    #mapping between pred labels and ground truth labels
    label_mapping = {}
    for louvain_label, gt_label in zip(louvain_labels, gt_labels):
        if louvain_label not in label_mapping:
            label_mapping[louvain_label] = gt_label
    
    
    louvain_labels_mapped = [label_mapping[label] for label in louvain_labels] #convert pred labels to ground truth labels

    accuracy = np.mean([1 if louvain == gt else 0 for louvain, gt in zip(louvain_labels_mapped,gt_labels)])
    print(f"Accuracy of Louvain: {accuracy:.3f}")
    
    labels = sorted(set(gt_labels + louvain_labels_mapped)) #Create a sorted list of unique labels
    num_classes = len(labels)

    cm = np.zeros((num_classes, num_classes), dtype=int)

    # Count the number of occurrences of each (true, pred) label pair
    for true, pred in zip(gt_labels, louvain_labels_mapped):
        cm[labels.index(true)][labels.index(pred)] += 1

    confusion_matrix_df = pd.DataFrame(cm, index=unique_labels, columns=range(num_classes))
    print(confusion_matrix_df)

    num_communities = len(set(partition.values()))
    hue_start = 0.0
    saturation = 0.8
    value = 0.8

    colors = []
    for i in range(num_communities):
        hue = hue_start + (i / num_communities)
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
        color_hex = "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))
        colors.append(color_hex)

    community_numb = list(partition.values())
    for i, n in enumerate(G_reptile_attr.nodes()):
        G_reptile.nodes[n]['color'] = colors[community_numb[i]]

    #network, config = visualize(G_reptile)
    #
    #visualize(G_reptile)

    colors = []
    for i in range(len(unique_labels)):
        hue = hue_start + (i / len(unique_labels))
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
        color_hex = "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))
        colors.append(color_hex)
    print(G_reptile_attr)
    #print(unique_labels)
    print(f"Amount of unique familys in dataset: {len(unique_labels):.0f}")
    for i, n in enumerate(G_reptile_attr.nodes()):
        G_reptile.nodes[n]['color'] = colors[unique_labels[G_reptile_attr.nodes[n]["Class"]]]
    for j in unique_labels.keys():
        if j in ["Reptilia", "Mammalia", "Amphibia", "Arachnida", "Aves", "Actinopterygii", "Insecta"]:
            print("Class: "+j+" Hexcode: " + colors[unique_labels[j]])
    #network, config = visualize(G_reptile)

    print("test")