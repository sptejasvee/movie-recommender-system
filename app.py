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
    api_key = st.secrets["TMDB_API_KEY"]
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}&language=en-US"
    
    try:
        response = requests.get(url)
        data = response.json()
        
        if 'poster_path' in data and data['poster_path']:
            return "https://image.tmdb.org/t/p/w500/" + data['poster_path']
        else:
            return "https://via.placeholder.com/500x750?text=No+Poster+Available"
            
    except Exception as e:
        return "https://via.placeholder.com/500x750?text=Connection+Error"

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
        return [], [], [], []
        
    idx = indices[movie_lower]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # Grab the top 10 matches
    sim_scores = sim_scores[1:11]
    
    recommended_movies = []
    recommended_posters = []
    recommended_overviews = []
    recommended_ids = []
    
    for i in sim_scores:
        movie_idx = i[0]
        movie_id = movies.iloc[movie_idx].id
        
        # Append details
        recommended_movies.append(movies.iloc[movie_idx].title)
        recommended_overviews.append(movies.iloc[movie_idx].overview)
        recommended_posters.append(fetch_poster(movie_id))
        recommended_ids.append(movie_id)
        
    return recommended_movies, recommended_posters, recommended_overviews, recommended_ids

# --- Build the Website UI ---
st.title('🍿 Movie Recommender System')
st.markdown("Discover your next favorite movie based on what you already love.")

# Using index=None makes it behave like a true search bar instead of a dropdown
selected_movie = st.selectbox(
    'Type a movie name to search:',
    movies['title'].values,
    index=None,
    placeholder="Search for a movie..."
)

if st.button('Show Recommendations'):
    if selected_movie:
        with st.spinner("Finding the best matches..."):
            names, posters, overviews, ids = recommend(selected_movie)
            
            if not names:
                st.error("Whoops! We couldn't find that movie in our database.")
            else:
                st.success("Here are your top 10 recommendations:")
                
                # Create a 2x5 grid (2 rows, 5 columns) for better UI scaling
                for row in range(0, 10, 5):
                    cols = st.columns(5)
                    
                    for col_idx in range(5):
                        # Calculate the actual index in our recommendation lists (0 through 9)
                        item_idx = row + col_idx 
                        
                        with cols[col_idx]:
                            tmdb_link = f"https://www.themoviedb.org/movie/{ids[item_idx]}"
                            
                            # Make the poster a clickable link using Markdown
                            st.markdown(
                                f"[![Poster]({posters[item_idx]})]({tmdb_link})", 
                                unsafe_allow_html=True
                            )
                            
                            # Make the title a clickable link
                            st.markdown(f"**[{names[item_idx]}]({tmdb_link})**")
                            
                            # Truncate and show overview
                            overview_text = overviews[item_idx]
                            if len(overview_text) > 100:
                                overview_text = overview_text[:100] + "..."
                            st.caption(overview_text)
    else:
        st.warning("Please type or select a movie first!")