################################################################
# This script takes in a email subject and matches it with the #
# all the support web links we have scrapped from the website ##
# and gives back top 3 matching links as output, this narrows ##
# down the webpages that the bot looks for to answer the query #
################################################################

import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import ollama

class top_matching_links:
    def __init__(self,support_link_path):
        with open(support_link_path, 'r') as file:
            self.support_links = file.read().splitlines()
    
    def clean_text(self,text):
        cleaned_text = re.sub(r'https?://[^/]+/', '', text)
        cleaned_text = re.sub(r'\?.*$', '', cleaned_text)
        cleaned_text = (cleaned_text.replace('-', ' ')).strip()
        return cleaned_text
    
    
    def get_distance(self,embed_1, embed_2):
        embedding1_np = np.array(embed_1).reshape(1, -1)
        embedding2_np = np.array(embed_2).reshape(1, -1)

        distance = cosine_similarity(embedding1_np, embedding2_np)[0][0]
        return distance
    
    def get_matching_url(self,url_list, query):
        query_embed = ollama.embeddings(prompt=self.clean_text(query),
                                        model='llama3')['embedding']
        matching_url = {}
        for url in url_list:
            distance = self.get_distance(query_embed,
                                    ollama.embeddings(prompt=self.clean_text(url),
                                                    model='llama3')['embedding'])
            matching_url[url] = distance
        return sorted(matching_url.items(), key=lambda item: item[1], reverse=True)[:3]
    
    def get(self, query):
        links = self.get_matching_url(self.support_links, query)
        links = [links[i][0] for i in range(len(links))]
        return links