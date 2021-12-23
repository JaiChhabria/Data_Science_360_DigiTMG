# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 15:55:13 2021

@author: jaich
"""

#Problem Statement: -

# The Entertainment Company, which is an online movie watching platform, wants to improve its collection of movies and showcase those that are highly rated and recommend those movies
# =============================================================================
#  to its customer by their movie watching footprint. For this, the company has collected the data and shared it with you to provide some analytical insights and also to come up with a recommendation algorithm so that it can automate its process for effective recommendations. The ratings are between -9 and +9.
# =============================================================================
import pandas as pd

data = pd.read_csv('Entertainment.csv', encoding='utf-8')

data.shape
data.columns
data.Category


from sklearn.feature_extraction.text import TfidfVectorizer


tfidf = TfidfVectorizer(stop_words = 'english')

data['Category'].isnull().sum() # Tehre are no missing values

tdidf_matrix = tfidf.fit_transform(data.Category)
tdidf_matrix.shape


from sklearn.metrics.pairwise import linear_kernel

cosine_sim_matrix = linear_kernel(tdidf_matrix,tdidf_matrix)

data_index = pd.Series(data.index, index= data['Titles']).drop_duplicates()

data_id = data_index['Heat (1995)']
data_id

def get_recommendations(Titles, topN):    
    # topN = 10
    # Getting the movie index using its title 
    data_id = data_index[Titles]
    
    # Getting the pair wise similarity score for all the anime's with that 
    # anime
    cosine_scores = list(enumerate(cosine_sim_matrix[data_id]))
    
    # Sorting the cosine_similarity scores based on scores 
    cosine_scores = sorted(cosine_scores, key=lambda x:x[1], reverse = True)
    
    # Get the scores of top N most similar movies 
    cosine_scores_N = cosine_scores[0: topN +1]
    
    # Getting the movie index 
    data_idx  =  [i[0] for i in cosine_scores_N]
    data_scores =  [i[1] for i in cosine_scores_N]
    
    # Similar movies and scores
    data_similar_title= pd.DataFrame(columns=["Title", "Score"])
    data_similar_title["Titles"] = data.loc[data_idx, "Titles"]
    data_similar_title["Score"] = data_scores
    data_similar_title.reset_index(inplace = True)  
    # anime_similar_show.drop(["index"], axis=1, inplace=True)
    print (data_similar_title)

get_recommendations("Heat (1995)", topN= 5)

get_recommendations("Babe (1995)", topN=5)


















