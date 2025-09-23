import numpy as np
import pandas as pd
import difflib  # use to get close match
from sklearn.feature_extraction.text import TfidfVectorizer # to transform text data to numerical features, with this it'll be easier to find the cosine_similarity
from sklearn.metrics.pairwise import cosine_similarity # to find similiar values (like similar movies)

movies_data  = pd.read_csv('movies.csv')
print(movies_data)
# selecting the relevant features for recommendation
    #  selected_features = ['geners', 'director', 'cast' , 'tagline' ,'keywords']
selected_features = ['genres', 'original_title', 'overview', 'tagline', 'keywords']


# loop through by replacing the null values with null string
for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')
    
    # print(movies_data.columns)

# combine all the 5 selected featurs
combined_features = movies_data['genres']+' '+movies_data['keywords']+' '+movies_data['original_title']+' '+movies_data['tagline']+' '+movies_data['overview']    
    
# converting text to numberical values (feature vectors) using TfidfVectorizer()
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)
    
# getting the similarities scores using cosine similarity
similarity = cosine_similarity(feature_vectors)
    
# getting user input and matching it with the list of movies
    # save user input in a variable (movie_name)
    # save all the title of the movies in a list = [''].tolist()
movie_name = input('Input movie title: ')
list_of_titles = movies_data['title'].tolist()
    
# using: difflib.get_close_matches(movie_name, list_of_titles)
    # save it in a variable as (find_close_match = )
find_close_match = difflib.get_close_matches(movie_name, list_of_titles)

close_match = find_close_match[0]
print(close_match)

# find the index of the movie with title using; index_of_the_movie = close match and movie_dataset 
index_of_the_movie = movies_data[movies_data['title'] == close_match].index[0]
print(index_of_the_movie)

# loop in the list to get the list of similar movies score; list(enumerate(similarity[index_of_the_movie]))
similarity_score = list(enumerate(similarity[index_of_the_movie]))
# print(similarity_score)

# sorting the movies based on their similarity score, from highest to lowest
sorted_similar_movies = sorted(similarity_score, key= lambda x:x[1], reverse=True)

i = 1
for movie in sorted_similar_movies:
    index = movie[0]
    title_from_index = movies_data[movies_data.index == index]['title'].values[0]
    
    if(i < 31):
        print(i, '-', title_from_index)
        i+=1
    