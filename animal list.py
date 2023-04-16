import requests
from bs4 import BeautifulSoup
import re
from get_links import links_on_page
import networkx
from tqdm import tqdm
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

if __name__ == "__main__":
    edgelist_weights = {}
    animal_name = "Elephant"
    names = names_from_table()
    for name in tqdm(names.keys()):
        result = links_on_page(animal_name=name.replace("/wiki/", ""))
    
        for entry in result:
            if entry in names:
                pair = ("/wiki/"+name,entry)
                pair_inverted = (entry,"/wiki/"+name)
                if pair in edgelist_weights:
                    edgelist_weights[pair] += 1
                elif pair_inverted in edgelist_weights:
                    edgelist_weights[pair_inverted] = 1
                else:
                    edgelist_weights[pair] = 1