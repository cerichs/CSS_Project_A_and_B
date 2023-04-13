import requests
from bs4 import BeautifulSoup

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
for tr in t_rows[1:]:
    tds = tr.find_all('td')
    row = [td.text.replace("\n","") for td in tds]
    rows.append(row)                    
import pandas as pd
df_animals = pd.DataFrame(rows, columns=header)


# Print the list of animal names
print(df_animals)