import streamlit as st
import pickle
import pandas as pd
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- UI Configuration ---
st.set_page_config(page_title="Movie Recommender", layout="wide")

# --- Helper Function: Fetch Posters from TMDB ---
def fetch_poster(movie_id):
    api_key = "TMDB_API_KEY"
    
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}&language=en-US"
    try:
        response = requests.get(url)
        data = response.json()
        if 'poster_path' in data and data['poster_path']:
            return "https://image.tmdb.org/t/p/w500/" + data['poster_path']
    except:
        pass
    # Fallback image if TMDB fails or movie has no poster
    return "https://via.placeholder.com/500x750?text=No+Poster+Available"

# --- Load Data & Compute Similarity ---
@st.cache_data
def load_data():
    movie_dict = pickle.load(open('movie_dict.pkl', 'rb'))
    movies = pd.DataFrame(movie_dict)
    
    cv = CountVectorizer(stop_words='english')
    count_matrix = cv.fit_transform(movies['tags'])
    cosine_sim = cosine_similarity(count_matrix, count_matrix)
    
    return movies, cosine_sim

movies, cosine_sim = load_data()

# --- Recommendation Engine ---
def recommend(movie):
    movie_lower = movie.lower()
    indices = pd.Series(movies.index, index=movies['title'].str.lower()).drop_duplicates()
    
    if movie_lower not in indices:
        return [], [], []
        
    idx = indices[movie_lower]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6] # Grab the top 5
    
    recommended_movies = []
    recommended_posters = []
    recommended_overviews = []
    
    for i in sim_scores:
        movie_idx = i[0]
        movie_id = movies.iloc[movie_idx].id
        
        # Append details
        recommended_movies.append(movies.iloc[movie_idx].title)
        recommended_overviews.append(movies.iloc[movie_idx].overview)
        recommended_posters.append(fetch_poster(movie_id))
        
    return recommended_movies, recommended_posters, recommended_overviews

# --- Build the Website UI ---
st.title('🍿 Movie Recommender System')
st.markdown("Discover your next favorite movie based on what you already love.")

selected_movie = st.selectbox(
    'Search for a movie:',
    movies['title'].values
)

if st.button('Show Recommendations'):
    with st.spinner("Finding the best matches..."):
        names, posters, overviews = recommend(selected_movie)
        
        if not names:
            st.error("Whoops! We couldn't find that movie in our database.")
        else:
            st.success("Here are your top 5 recommendations:")
            
            # Create a 5-column layout for a modern, Netflix-style look
            cols = st.columns(5)
            
            for i in range(5):
                with cols[i]:
                    # Show Poster
                    st.image(posters[i], use_container_width=True)
                    
                    # Show Title
                    st.subheader(names[i])
                    
                    # Show Overview (Truncated so the UI stays neat)
                    overview_text = overviews[i]
                    if len(overview_text) > 120:
                        overview_text = overview_text[:120] + "..."
                    st.caption(overview_text)