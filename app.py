import streamlit as st
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Load the data
movie_dict = pickle.load(open('movie_dict.pkl', 'rb'))
movies = pd.DataFrame(movie_dict)

# 2. Cache the heavy computation so it only runs once when the app boots
@st.cache_data
def calculate_similarity(data):
    cv = CountVectorizer(stop_words='english')
    count_matrix = cv.fit_transform(data['tags'])
    cosine_sim = cosine_similarity(count_matrix, count_matrix)
    return cosine_sim

cosine_sim = calculate_similarity(movies)

# 3. Recommendation Function
def recommend(movie):
    movie_lower = movie.lower()
    
    # Create a temporary index series just like in your notebook
    indices = pd.Series(movies.index, index=movies['title'].str.lower()).drop_duplicates()
    
    if movie_lower not in indices:
        return ["Movie not found in database."]
        
    idx = indices[movie_lower]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11] # Get top 10
    
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices].values

# 4. Build the Website UI
st.title('🎬 Movie Recommender System')

selected_movie = st.selectbox(
    'Type or select a movie to get recommendations:',
    movies['title'].values
)

if st.button('Recommend'):
    recommendations = recommend(selected_movie)
    for i in recommendations:
        st.write(i)