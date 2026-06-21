import streamlit as st
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.stem.porter import PorterStemmer
import requests

# Page width ko wide karna takki 5 posters side-by-side aa sakein
st.set_page_config(layout="wide")

try:
    ps = PorterStemmer()
except Exception:
    nltk.download('punkt')
    ps = PorterStemmer()

# 1. Load movies data
movies = pickle.load(open('movies.pkl', 'rb'))

# 2. Runtime par similarity matrix banana
@st.cache_data
def calculate_similarity(_df):
    stemmed_tags = _df['tags'].astype(str).apply(lambda text: " ".join([ps.stem(i) for i in text.split()]))
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(stemmed_tags).toarray()
    similarity_matrix = cosine_similarity(vectors)
    return similarity_matrix

similarity = calculate_similarity(movies)

# 3. TMDB API se poster fetch karne ka function
def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US"
    try:
        data = requests.get(url, timeout=5).json()
        poster_path = data['poster_path']
        full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
        return full_path
    except Exception:
        # Agar poster na mile to blank placeholder image
        return "https://via.placeholder.com/500x750?text=No+Poster"

# 4. Recommend function jo naam aur poster dono return karega
def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    
    recommended_movies = []
    recommended_movies_posters = []
    
    for i in movies_list:
        # movie_id nikalna poster ke liye
        movie_id = movies.iloc[i[0]].movie_id 
        
        recommended_movies.append(movies.iloc[i[0]].title)
        recommended_movies_posters.append(fetch_poster(movie_id))
        
    return recommended_movies, recommended_movies_posters

# --- Streamlit UI ---
st.title("🎬 Movie Recommendation System")

selected_movie = st.selectbox(
    "Select a Movie",
    movies['title'].values
)

if st.button("Recommend"):
    names, posters = recommend(selected_movie)
    
    st.subheader("Top 5 Recommended Movies")
    
    # 5 Columns banana side-by-side dikhane ke liye
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.text(names[0])
        st.image(posters[0])
    with col2:
        st.text(names[1])
        st.image(posters[1])
    with col3:
        st.text(names[2])
        st.image(posters[2])
    with col4:
        st.text(names[3])
        st.image(posters[3])
    with col5:
        st.text(names[4])
        st.image(posters[4])