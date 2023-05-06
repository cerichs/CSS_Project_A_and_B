import networkx as nx
import pickle
import random
import community
import matplotlib.pyplot as plt


def compute_modularity(Graph,partitioning):
    L = Graph.number_of_edges()
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
    while(i<2*N):
        edges = list(G_copy.edges()) # update edges after adding and removal
        (u, v), (x, y) = random.sample(edges, 2) # picking two random edges
        if (u != v) and (v != x) and (u, y) not in G_copy.edges() and (x, v) not in G_copy.edges(): # checking conditions
            G_copy.add_edge(u, y)
            G_copy.add_edge(x, v)
            G_copy.remove_edge(u, v)
            G_copy.remove_edge(x, y)
            i+=1
    return G_copy

if __name__ == "__main__":
    with open('data_plain_long_reptile.pickle', 'rb') as handle:
        b = pickle.load(handle)
    edgelist = [None]*len(b)
    for i,items in enumerate(b):
        edgelist[i] = (items[0].replace("/wiki/", ""),items[1].replace("/wiki/", ""),int(b[items]))
    G_reptile = nx.Graph()

    #for entries in edgelist:
    G_reptile.add_weighted_edges_from(edgelist)
    print(G_reptile)

    G_new = double_edge_swap(G_reptile,len(list(G_reptile.edges())))
    degree_1 = []
    for node in G_reptile.nodes():
        degree_1.append(G_reptile.degree(node))
    degree_2 = []
    for node in G_new.nodes():
        degree_2.append(G_new.degree(node))

    print(degree_1)
    print(degree_2)
    print(degree_1 == degree_2)
    plt.hist(degree_1,bins=20)
    plt.title("Degree for Reptile Network")
    plt.show()
    plt.hist(degree_2,bins=20)
    plt.title("Degree for new Network")
    plt.show()
    
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