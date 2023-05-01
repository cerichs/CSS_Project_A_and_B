# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 09:03:55 2023

@author: Cornelius
"""

import requests
import json
from bs4 import BeautifulSoup
def links_on_page(animal_name="Elephant"):
    url = "https://en.wikipedia.org/w/api.php?action=parse&page="+animal_name+"&format=json"
    response = requests.get(url)
    html = json.loads(response.content.decode('utf-8'))['parse']['text']['*']
    soup = BeautifulSoup(html, 'html.parser')
    links = soup.find_all('a', href=lambda href: href and href.startswith('/wiki/') and not href.endswith('.jpg') and not href.endswith('.png'))
    result = []
    for link in links:
        href = link.get('href')
        title = link.get('title')
        text = link.text
        if False:
            # Sometimes a link is just a redirect to a different page
            redirect_url = f"https://en.wikipedia.org{href}"
            redirect_response = requests.get(redirect_url)
            redirect_html = redirect_response.content.decode('utf-8')
            
            if "#REDIRECT" in redirect_html:
                redirect_soup = BeautifulSoup(redirect_html, 'html.parser')
                new_title = redirect_soup.find('link', {'rel': 'canonical'}).get('title') #not all pages contain redirect div
                if new_title is None:
                    new_title = redirect_soup.find('title').text.replace(' - Wikipedia', '')
                #print(f"Link: {href}\nRedirects to: {new_title}\nText: {text}\n")
            #else:
                #print(f"Link: {href}\nTitle: {title}\nText: {text}\n")
    
        result.append(href)
    return result

if __name__ == "__main__":
    lists = links_on_page(animal_name="Elephant")
    