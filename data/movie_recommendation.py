import numpy as np
import pandas as pd
import difflib  # use to get close match
from sklearn.feature_extraction.text import TfidfVectorizer # to transform text data to numerical features, with this it'll be easier to find the cosine_similarity
from sklearn.metrics.pairwise import cosine_similarity # to find similiar values (like similar movies)

movies_data  = pd.read_csv('movies.csv')
print(movies_data.columns)

# selecting the relevant features for recommendation
    #  selected_features = ['geners', 'director', 'cast' , 'tagline' ,'keywords']

# loop through by replacing the null values with null string
    # ....fillna('')
    
# combine all the 5 selected featurs
    # combined_data = []
    
# converting text to numberical values (feature vectors) using TfidfVectorizer()
    # vectorizer = TfidfVectorizer()
    # feature_vectors = vectorizer.fit_transform(combined_data)
    
# getting the similarities scores using cosine similarity
    # similarity = cosine_similarity(feature_vectors)
    
# getting user input and matching it with the list of movies
    # save user input in a variable (movie_name)
    # save all the title of the movies in a list = [''].tolist()
    
# using: difflib.get_close_matches(movie_name, list_of_titles)
    # save it in a variable as (find_close_match = )

# close_match = find close match[0]

# find the index of the movie with title using. index_of_the_movie = close match and movie_dataset 

# loop in the list to get the list of similar movies score; list(enumerate(similarity[index_of_the_movie]))