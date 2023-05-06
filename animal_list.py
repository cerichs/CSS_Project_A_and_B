import requests
from bs4 import BeautifulSoup
import re
from get_links import links_on_page
import networkx as nx
from tqdm import tqdm
import matplotlib.pyplot as plt
#from netwulf import visualize
import json
import pickle

def names_from_table():
    url = "https://en.wikipedia.org/wiki/List_of_animal_names"
    
    response = requests.get(url)
    html_content = response.content
    
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Find table on the page
    table = soup.find_all('table',{"class":"wikitable"})[1]
    
    # Find all the rows in the table
    t_rows = table.find_all('tr')
    
    ths = t_rows[0].find_all('th')
    
    header = [th.text.replace("\n","") for th in ths]
    
    rows = []
    names = {}
    for tr in t_rows[1:]:
        tds = tr.find_all('td')
        #row = [td.text.replace("\n","") for td in tds]
        if tds:
            links = tds[0].find_all('a')
            if links:
                row = [td.text.replace("\n","") for td in tds]
                rows.append(row)
                cleaned_row = re.sub(r'\(.*\)|Also see.*|\[\d+\]|See.*', '', row[0]) # The table is not clean, many unwanted formating we remove here
                #cleaned_row = re.sub(r'\[\d+\]', '', row[0])
                #names[cleaned_row] = 0
                for link in links:
                    link_href = link.get('href')
                    if link_href.startswith('/wiki/'):
                        #link_href = 'https://en.wikipedia.org' + link_href
                        names[link_href] = cleaned_row
                        break
    import pandas as pd
    df_animals = pd.DataFrame(rows, columns=header)

    return names

def is_redirect(name):
    url = f'https://en.wikipedia.org/w/api.php?action=query&titles={name}&redirects=1&format=json'
    response = requests.get(url)
    data = response.json()
    if "redirects" in data["query"]:
        return data["query"]["redirects"][0]["to"]
    else: 
        return name
    
    
if __name__ == "__main__":
    edgelist_weights = {}
    edgelist_weights_long = {}
    #animal_name = "Elephant"
    names = names_from_table() # Gets all the entries in the larger table on https://en.wikipedia.org/wiki/List_of_animal_names
    names_long = {}
    with open('data/animal_links.txt', 'r') as f:
        entries = f.read().splitlines()
    for name in tqdm(entries):
        name_temp = name.split("/")[-1]
        names_long[name_temp] = 0 # Saving all the entries in the txt file in a dict, for a fast comparisons (ie. only make pairs
                                  #  with animals and not wikipages for unrelated stuff)
    attributes_dict = {}
    for name in tqdm(entries):
        temp_string = name.split("/")[-1]
        #cleaned_name = name.replace("/wiki/", "")
        result, info = links_on_page(animal_name=temp_string)
        if info["Name:"] is not None: # A way to remove wiki redirects from the final result, as redirects dont have the infoboxes we want
            attributes_dict[info["Name:"]] = info # Making nested dicts to quickly get attributes later
            for entry in result:
                #entry = name if is_redirect(name) != name else entry
                if entry in names: # making one graph where we only make edges to the table from https://en.wikipedia.org/wiki/List_of_animal_names
                    pair = ("/wiki/"+temp_string,entry) # Making pairs to compare for the dict
                    pair_inverted = (entry,"/wiki/"+temp_string)
                    if pair in edgelist_weights:
                        edgelist_weights[pair] += 1 # If the pair is already in the dict, the weight is increased
                    elif pair_inverted in edgelist_weights:
                        edgelist_weights[pair_inverted] += 1 # If the inverted pair is already in the dict, the weight is increased
                    else:
                        edgelist_weights[pair] = 1 # If the pair is not in the dict, the weight is 1
                if entry.split("/")[-1] in names_long: # making other graph where we make edges between entries from the .txt file
                    pair = (temp_string, entry.split("/")[-1]) # Only taking the last part of the URL (the URL title of the page)
                    if pair in edgelist_weights_long:
                        edgelist_weights_long[pair] += 1 # If the pair is already in the dict, the weight is increased
                    else:
                        edgelist_weights_long[pair] = 1 # If the pair is not in the dict, the weight is 1
    with open('data/all_animal_to_animal_list.pickle', 'wb') as fp:
        pickle.dump(edgelist_weights, fp, protocol=pickle.HIGHEST_PROTOCOL)
    with open('data/all_animal_to_all_animal.pickle', 'wb') as fp:
        pickle.dump(edgelist_weights_long, fp, protocol=pickle.HIGHEST_PROTOCOL)
    with open('data/animal_attributes.pickle', 'wb') as fp:
        pickle.dump(attributes_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)


    values = list(edgelist_weights.values())
    plt.hist(values, bins=max(values), edgecolor='black')
    plt.xticks(range(1, max(values) + 1))
    plt.xlabel('Number of references')
    plt.ylabel('Frequency')
    plt.show()
    
    edgelist = [None]*len(edgelist_weights)
    for i,items in enumerate(edgelist_weights):
        edgelist[i] = (items[0].replace("/wiki/", ""),items[1].replace("/wiki/", ""),int(edgelist_weights[items]))
    G = nx.Graph()

    #for entries in edgelist:
    G.add_weighted_edges_from(edgelist)
    print(G)
    to_remove =[]
    for Names in tqdm(G.nodes):
        if Names in attributes_dict:
            G.nodes[Names]['Class'], G.nodes[Names]['Order'], G.nodes[Names]['Superfamily'], G.nodes[Names]['Family'], _ = attributes_dict[Names].values()
        else:
            to_remove.append(Names) # Some nodes get added to graph even though they are redirects, the cause is known but no good way to handle it
    for names in to_remove:
        G.remove_node(names)
    data1 = nx.node_link_data(G)
    json.dump(data1, open('data/data_total_all_animal.json','w'))
    #network, config = visualize(G)
