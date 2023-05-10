"""
Copyright 2023 Nitya

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

#!/usr/bin/env python
# coding: utf-8

# In[7]:


from IPython import get_ipython
ipython_shell = get_ipython()

# In[ ]:


get_ipython().system(u'pip install beautifulsoup4')
get_ipython().system(u'pip install google')
get_ipython().system(u'pip install nltk')
get_ipython().system(u'pip install pandas')



# In[10]:



from googlesearch import search
topic = input("Enter a topic you want to explore: ")

top10 = []
exclusion_list = []

for i in search(topic, tld = "com", num=10, stop=10, pause=2):
    print(i)
    top10.append(i)

exclude = input("Which searches should we exclude from the top 10 searches listed above?  ")


to_list = exclude.split(',')

# exclusion_list.append(map(int,str(list)))
# print(exclusion_list)
for i in to_list:
    exclusion_list.append(int(i))

for j in sorted(exclusion_list, reverse=True):
    del top10[j-1]

print(top10)


# In[11]:


from bs4 import BeautifulSoup
import json
import numpy as np
import requests
from requests.models import MissingSchema
import spacy
import trafilatura

data = {}

for url in top10:
    # 1. Obtain the response:
    resp = requests.get(url)
    
    # 2. If the response content is 200 - Status Ok, Save The HTML Content:
    if resp.status_code == 200:
        data[url] = resp.text

def beautifulsoup_extract_text_fallback(response_content):
    
    '''
    This is a fallback function, so that we can always return a value for text content.
    Even for when both Trafilatura and BeautifulSoup are unable to extract the text from a 
    single URL.
    '''
    
    # Create the beautifulsoup object:
    soup = BeautifulSoup(response_content, 'html.parser')
    
    # Finding the text:
    text = soup.find_all(text=True)
    
    # Remove unwanted tag elements:
    cleaned_text = ''
    blacklist = [
        '[document]',
        'noscript',
        'header',
        'html',
        'meta',
        'head', 
        'input',
        'script',
        'style',]

    # Then we will loop over every item in the extract text and make sure that the beautifulsoup4 tag
    # is NOT in the blacklist
    for item in text:
        if item.parent.name not in blacklist:
            cleaned_text += '{} '.format(item)
            
    # Remove any tab separation and strip the text:
    cleaned_text = cleaned_text.replace('\t', '')
    return cleaned_text.strip()

def extract_text_from_single_web_page(url):
    
    downloaded_url = trafilatura.fetch_url(url)
    try:
        a = trafilatura.extract(downloaded_url, output_format = 'json', with_metadata=True, include_comments = False,
                            date_extraction_params={'extensive_search': True, 'original_date': True})
    except AttributeError:
        a = trafilatura.extract(downloaded_url, output_format = 'json', with_metadata=True,
                            date_extraction_params={'extensive_search': True, 'original_date': True})
    if a:
        json_output = json.loads(a)
        return json_output['text']
    else:
        try:
            resp = requests.get(url)
            # We will only extract the text from successful requests:
            if resp.status_code == 200:
                return beautifulsoup_extract_text_fallback(resp.content)
            else:
                # This line will 
                #handle for any failures in both the Trafilature and BeautifulSoup4 functions:
                return np.nan
        # Handling for any URLs that don't have the correct protocol
        except MissingSchema:
            return np.nan


#     html_page = res.content
#     soup = BeautifulSoup(html_page, 'html.parser')
#     text = soup.find_all(string=True)

# single_url = 'https://www.interviewbit.com/pyspark-interview-questions/'
# text = extract_text_from_single_web_page(single_url)
#print(text)
for i in range(len(top10)):
    url = top10[i]
    j=str(i+1)
    text = extract_text_from_single_web_page(url)
    filename = 'url'+ j +'.txt'
    with open(filename, 'w') as file:
        file.write(str(text))


# In[12]:


data = ""
for i in range(len(top10)):
    j=str(i+1)
    filename = 'url' + j + '.txt'
    with open(filename) as fs:
        summary_data = fs.read()
    data = data + summary_data
    
with open('final.txt', 'w') as file:
    file.write(str(data))
    
    


# In[13]:


import openai
openai.api_key = "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"


# In[14]:


def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]


# In[15]:


import os
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize


# In[16]:


def count_tokens(filename):
    with open(filename, 'r') as f:
        text = f.read()
    tokens = word_tokenize(text)
    return len(tokens)


filename = 'final.txt'
token_count = count_tokens(filename)
print(f"Number of tokens: {token_count}")


# In[17]:



def break_up_file(tokens, chunk_size, overlap_size):
    if len(tokens) <= chunk_size:
        yield tokens
    else:
        chunk = tokens[:chunk_size]
        yield chunk
        yield from break_up_file(tokens[chunk_size-overlap_size:], chunk_size, overlap_size)
        

def break_up_file_to_chunks(filename, chunk_size=2000, overlap_size=100):
    with open(filename, 'r') as f:
        text = f.read()
    tokens = word_tokenize(text)
    split = [*break_up_file(tokens, chunk_size, overlap_size)]
    return split

def convert_to_detokenized_text(tokenized_text):
    prompt_text = " ".join(tokenized_text)
    prompt_text = prompt_text.replace(" 's", "'s")
    return prompt_text



filename = 'final.txt'

chunks = break_up_file_to_chunks(filename)
for i, chunk in enumerate(chunks):
    print(f"Chunk {i}: {len(chunk)} tokens")


# In[18]:


filename = "final.txt"

prompt_response = []
chunks = break_up_file_to_chunks(filename)

for i, chunk in enumerate(chunks):
    prompt = f"""
            Summarize the questions below, delimited by triple \
            backticks focussing on not duplicating the questions. \

            Questions: ```{convert_to_detokenized_text(chunks[i])}``` 
            """ 
    response = get_completion(prompt)
    
    prompt_response.append(response.strip())
print(prompt_response)


# In[ ]:




