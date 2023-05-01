# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 09:15:04 2023

@author: Cornelius
"""
from tqdm import tqdm
import requests
import json

def get_wiki_links(item_ids):
    url = f"https://www.wikidata.org/w/api.php?action=wbgetentities&ids={'|'.join(item_ids)}&props=sitelinks/urls&format=json&sitefilter=enwiki"
    response = requests.get(url)
    data = json.loads(response.text)
    wiki_links = []
    for item_id in item_ids:
        wikipedia_url = data["entities"][item_id]["sitelinks"]
        if "enwiki" in wikipedia_url: # only save if the page has a reference to an english wikipage
            link = wikipedia_url["enwiki"]["url"]
            wiki_links.append(link)
    return wiki_links

if __name__ == "__main__":
    url = 'https://query.wikidata.org/sparql'
    query = '''
    SELECT DISTINCT ?item ?itemLabel WHERE {
      SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE]". }
      {
        SELECT DISTINCT ?item WHERE {
          ?item p:P4024 ?statement0.
          ?statement0 (ps:P4024) _:anyValueP4024.
        }
      }
    }
    '''
    r = requests.get(url, params = {'format': 'json', 'query': query})
    data = r.json()
    
    data = data["results"]["bindings"]
    temp = []
    for entries in tqdm(range(0,len(data),50)):
        if entries+50 < len(data):
            sub_list = [ids["itemLabel"]["value"] for ids in data[entries:(entries+50)]]
            temp = temp + get_wiki_links(sub_list)
        else:
            sub_list = [ids["itemLabel"]["value"] for ids in data[entries:(len(data))]]
            temp = temp + get_wiki_links(sub_list)
    file = open('animal_links.txt','w')
    for item in temp:
        file.write(item+"\n")
    file.close()
