import pandas as pd
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ✅ Load Dataset
@st.cache_data
def load_data():
    df = pd.read_csv(r"C:/Users/abinaya/Desktop/sentimental analysis/imdb4.0.csv")  # Update with actual file name
    df = df[['Series_Title', 'Genre', 'Overview']].dropna()  # Keep necessary columns
    return df

movies = load_data()

# ✅ Convert Text into Numerical Features (TF-IDF)
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['Overview'])

# ✅ Compute Similarity Matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# ✅ Recommendation Function
def get_recommendations(title):
    idx = movies[movies['Series_Title'] == title].index
    if len(idx) == 0:
        return ["❌ Movie not found."]
    
    idx = idx[0]  # Get the first index
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return movies['Series_Title'].iloc[movie_indices].tolist()

# ✅ Streamlit UI
st.title("🎬 Movie Recommendation System")
st.write("Enter a movie name and get similar movie recommendations!")

# User Input
movie_name = st.selectbox("Select or type a movie:", movies['Series_Title'].tolist())

if st.button("Get Recommendations"):
    recommendations = get_recommendations(movie_name)
    st.subheader("📌 Recommended Movies:")
    for i, movie in enumerate(recommendations):
        st.write(f"**{i+1}. {movie}**")
