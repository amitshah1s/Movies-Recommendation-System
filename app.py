import streamlit as st
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.stem.porter import PorterStemmer

# NLTK data ensure karna download ho jaye
try:
    ps = PorterStemmer()
except Exception:
    nltk.download('punkt')
    ps = PorterStemmer()

# 1. Load movies data (Jo aapka new_df hai)
movies = pickle.load(open('movies.pkl', 'rb'))

# Helper function stemming ke liye
def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

# 2. Runtime par similarity matrix generate karne ka cache function
@st.cache_data
def calculate_similarity(_df):
    # Tags ko ensure karna ki wo string format me ho aur apply karein stemming
    # (Kyunki notebook me humne lowercase aur join pehle hi kar diya tha)
    stemmed_tags = _df['tags'].astype(str).apply(stem)
    
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(stemmed_tags).toarray()
    
    similarity_matrix = cosine_similarity(vectors)
    return similarity_matrix

# Similarity matrix calculate/load karna
st.text("Loading recommendations engine...")
similarity = calculate_similarity(movies)

def recommend(movie):
    # Exact notebook wala index extraction
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    
    movies_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]
    
    recommended_movies = []
    for i in movies_list:
        recommended_movies.append(movies.iloc[i[0]].title)
        
    return recommended_movies

# --- Streamlit UI ---
st.title("🎬 Movie Recommendation System")

selected_movie = st.selectbox(
    "Select a Movie",
    movies['title'].values
)

if st.button("Recommend"):
    recommendations = recommend(selected_movie)
    
    st.subheader("Recommended Movies")
    for movie in recommendations:
        st.write("✅", movie)