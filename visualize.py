import requests
from bs4 import BeautifulSoup
import re
from get_links import links_on_page
import networkx as nx
from tqdm import tqdm
import matplotlib.pyplot as plt
from netwulf import visualize
import json
import pickle


if __name__ == "__main__":
    with open('data/data_plain_reptile.pickle', 'rb') as handle:
        a = pickle.load(handle)
    with open('data/data_plain_long_reptile.pickle', 'rb') as handle:
        b = pickle.load(handle)
    with open('data/Reptile_attributes.pickle', 'rb') as handle:
        c = pickle.load(handle)
    
    values = list(b.values())
    plt.hist(values, bins=max(values), edgecolor='black')
    plt.xticks(range(1, max(values) + 1))
    plt.xlabel('Number of references')
    plt.ylabel('Frequency')
    plt.show()
    
    edgelist = [None]*len(b)
    for i,items in enumerate(b):
        edgelist[i] = (items[0].replace("/wiki/", ""),items[1].replace("/wiki/", ""),int(b[items]))
    G = nx.DiGraph()

    #for entries in edgelist:
    G.add_weighted_edges_from(edgelist)
    print(G)
    #network, config = visualize(G)
    to_remove =[]
    for Names in tqdm(G.nodes):
        if Names in c:
            G.nodes[Names]['Class'], G.nodes[Names]['Order'], G.nodes[Names]['Superfamily'], G.nodes[Names]['Family'], _ = c[Names].values()
        else:
            to_remove.append(Names) # Some nodes get added to graph even though they are redirects, the cause is known but no good way to handle it
    for names in to_remove:
        G.remove_node(names)
    data1 = nx.node_link_data(G)
    json.dump(data1, open('data/data_total.json','w'))
    