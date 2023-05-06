# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 09:03:55 2023

@author: Cornelius
"""

import requests
import json
from bs4 import BeautifulSoup
def links_on_page(animal_name="Elephant"):
    url = "https://en.wikipedia.org/w/api.php?action=parse&page="+animal_name+"&format=json" # we can either put the title page or the URL version
                                                                                             # of the title, ex. Malayan softshell turtle = Malayan_softshell_turtle
    response = requests.get(url)
    html = json.loads(response.content.decode('utf-8'))['parse']['text']['*'] # Default way the result comes in
    soup = BeautifulSoup(html, 'html.parser')
    links = soup.find_all('a', href=lambda href: href and href.startswith('/wiki/') and not href.endswith('.jpg') and not href.endswith('.png'))
    # ^ sometimes the links link to jpg and png file, sort them out
    result = []
    for link in links:
        href = link.get('href')
        title = link.get('title')
        text = link.text
        result.append(href)
    info = {"Class:": None, "Order:":None, "Superfamily:":None, "Family:":None,"Name:": None} # Default dict allows us to see if its redirect later
    infobox = soup.find('table', {'class': 'infobox biota biota-infobox'}) # One type of infoboxes wiki uses
    if infobox is None:
        infobox = soup.find('table', {'class': 'infobox biota'}) # Other type of infoboxes wiki uses
    if infobox:
        rows = infobox.find_all('tr') # Going through the rows in the infobox
        for row in rows:
            td = row.find_all('td')
            if td: # Checking if empty
                if td[0].text.strip() in ["Class:", "Order:", "Superfamily:", "Family:"]: # the info we want is stored in a row at a time
                                                                                          # it has two columns first being Class, Order...
                                                                                          # the other being the attributes we will save 
                    if td:
                        info[td[0].text.strip()] = td[1].text.strip() # Making the category the key, and the attribute value the value
        info["Name:"] = animal_name # Updating from default 

    return result, info

if __name__ == "__main__":
    lists, info = links_on_page(animal_name="Aplopeltura_boa")
    