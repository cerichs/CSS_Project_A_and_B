# %%
# from get_links import links_on_page
import numpy as np
import pandas as pd
import requests
import re
import networkx as nx
import matplotlib.pyplot as plt
import nltk
import sklearn
from bs4 import BeautifulSoup
from tqdm import tqdm
tqdm.pandas()
from animal_list import names_from_table
from netwulf import visualize

# %%
animal_names = names_from_table()
# read txt file with pandas
animal_df = pd.read_csv('animal_links.txt', header=None)
animal_df.columns = ['page-name']

# remove the first part of the url
animal_df['page-name'] = animal_df['page-name'].str.replace('https://en.wikipedia.org', '')
animal_df["name"] = animal_df["page-name"].str.split("/").str[-1]


# %%
# animal_names
animal_df

# %%
# create function that will run on each row in the dataframe, which will take the page name and return all the readable text on the page

def get_text(page_name):
    url = 'https://en.wikipedia.org' + page_name
    response = requests.get(url)
    html_content = response.content
    soup = BeautifulSoup(html_content, 'html.parser')
    # Find all the paragraphs in the body
    paragraphs = soup.body.find_all('p')
    # Extract the text from the paragraphs but remove \n
    # text = [p.text for p in paragraphs]
    text = [str(p.text.replace('\n', '')).strip() for p in paragraphs]
    # Join the paragraphs together
    joined_text = ' '.join(text)
    # remove first space
    joined_text = joined_text[1:]
    return joined_text
        

# %%
get_text('/wiki/Aardvark')

# %%
# create a new column in the dataframe that will contain the text from the page
animal_df['text'] = animal_df['page-name'].progress_apply(get_text)

# %%
def tokenize(text):
    # convert to lowercase
    text = text.lower()

    # remove punctuation
    text = re.sub(r'[^\w\s]', '', text)

    # remove urls
    text = re.sub(r'http\S+', '', text)

    # remove numbers
    text = re.sub(r'\d+', '', text)

    # tokenize
    tokens = nltk.word_tokenize(text)

    # remove stopwords
    stop_words = set(nltk.corpus.stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    return tokens

# %%
# create a new column in the dataframe that will contain the tokenized text from the page
animal_df['tokens'] = animal_df['text'].apply(tokenize)

# %%
animal_df

# %%
# td-idf
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(animal_df['text'])
tfidf_matrix.shape



# %%
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    return text

# %%
vectorizer = TfidfVectorizer(stop_words='english', preprocessor=preprocess_text)
# vectorizer = TfidfVectorizer(tokenizer=tokenize, stop_words='english')
vectors = vectorizer.fit_transform(animal_df['text'])
feature_names = vectorizer.get_feature_names_out()
dense = vectors.todense()
denselist = dense.tolist()
df = pd.DataFrame(denselist, columns=feature_names)

# %%
def tf_idf(corpus):
    tf_dist = [nltk.FreqDist(text) for tokens in corpus]



