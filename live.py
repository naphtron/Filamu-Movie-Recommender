import streamlit as st
import pandas as pd
import pickle 
import requests
import numpy as np

###
movie_rating_df = pd.read_csv('ml-latest-small/movie_rating_df.csv')
movies = pd.read_csv('ml-latest-small/movies.csv')
movies = movies.sample(frac=1, random_state=42).reset_index(drop=True)



# movie = movies[movies['genres'].str.contains('Adventure')].sample(1)
# print(movie)

movie_mapper = dict(zip(movies['title'], movies.index))
user_item_df = movie_rating_df[['userId', 'movieId', 'rating']]
ratings = [1,2,3,4,5,'n']
genres = ['Adventure',
         'Animation',
         'Children',
         'Comedy',
         'Fantasy',
         'Romance',
         'Drama',
         'Action',
         'Crime',
         'Thriller',
         'Horror',
         'Mystery',
         'Sci-Fi',
         'War',
         'Musical',
         'Documentary',
         'IMAX',
         'Western',
         'Film-Noir',
         '(no genres listed)']
content_matrix = pickle.load(open('content_matrix.sav','rb'))
from sklearn.metrics.pairwise import cosine_similarity

def get_movie_recommendations(movie, n):

    M_idx = movie_mapper[movie]
    # print('Movie Index: ',M_idx)
    movie_index = content_matrix[M_idx]
    # print('Index: ', type(movie_index))
    scores = cosine_similarity(movie_index, content_matrix)
    scores = scores.flatten()
    print('Scores: ',scores)

    recommended_indices = (-scores).argsort()[1:n+1]
    print("Rec: ",recommended_indices)
    return movies['title'].iloc[recommended_indices],movies['movieId'].iloc[recommended_indices]
##
def fetch_poster(movie_id):
    response = requests.get('https://api.themoviedb.org/3/movie/{}?api_key=e62e106a7409881e60db5a8aaa58e0a5&language=en-US'.format(movie_id))
    data = response.json()
    st.text(data)
    return "https://image.tmdb.org/t/p/w500" + data['poster_path']

st.title('Filamu Movie Recommender')

selected_movie_name = st.selectbox("Which movie is similar to what you'd wanna watch?",movies['title'].values)
if st.button('Search'):
    names, indices = get_movie_recommendations(selected_movie_name, 10)
    for name,id in zip(names, indices):
        # st.text(id)
        # st.table(name)
        st.text(name)
        # st.text(name)
        # st.image(fetch_poster(id))
    # for name in names:
    #     for id in idx:
    #         st.image(fetch_poster(id))
    #         st.text(name)
