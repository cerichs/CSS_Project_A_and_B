import requests
from bs4 import BeautifulSoup
import re
from get_links import links_on_page
import networkx as nx
from tqdm import tqdm
import matplotlib.pyplot as plt
#from netwulf import visualize
import json

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
                cleaned_row = re.sub(r'\(.*\)|Also see.*|\[\d+\]|See.*', '', row[0])
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
    names = names_from_table()
    names_long = {}
    with open('animal_links.txt', 'r') as f:
        entries = f.read().splitlines()
    for name in tqdm(entries):
        name_temp = name.split("/")[-1]
        names_long[name_temp] = 0
    
    for name in tqdm(entries):
        temp_string = name.split("/")[-1]
        #cleaned_name = name.replace("/wiki/", "")
        result = links_on_page(animal_name=temp_string)
    
        for entry in result:
            #entry = name if is_redirect(name) != name else entry
            if entry in names:
                pair = ("/wiki/"+temp_string,entry)
                pair_inverted = (entry,"/wiki/"+temp_string)
                if pair in edgelist_weights:
                    edgelist_weights[pair] += 1
                elif pair_inverted in edgelist_weights:
                    edgelist_weights[pair_inverted] += 1
                else:
                    edgelist_weights[pair] = 1
            if entry.split("/")[-1] in names_long:
                pair = (temp_string, entry.split("/")[-1])
                if pair in edgelist_weights_long:
                    edgelist_weights_long[pair] += 1
                else:
                    edgelist_weights_long[pair] = 1
    with open('data_plain.json', 'w') as fp:
        json.dump(edgelist_weights, fp)
    with open('data_plain_long.json', 'w') as fp:
        json.dump(edgelist_weights_long, fp)
    with open('data_pretty.json', 'w') as fp:
        json.dump(edgelist_weights, fp, sort_keys=True, indent=4)

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
    #network, config = visualize(G)
