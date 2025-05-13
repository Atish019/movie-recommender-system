import streamlit as st
import pandas as pd
import pickle

# Load data & model
movies = pd.read_csv("data/processed/movies_processed.csv")
with open("models/tfidf.pkl", "rb") as f:
    tfidf = pickle.load(f)
with open("models/movie_indices.pkl", "rb") as f:
    movie_indices = pickle.load(f)
with open("models/cosine_sim.pkl", "rb") as f:
    cosine_sim = pickle.load(f)

# Recommendation function (Optimized)
def get_recommendations(title):
    idx = movie_indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # Top 10 recommendations
    movie_indices_rec = [i[0] for i in sim_scores]
    return movies["title"].iloc[movie_indices_rec]

# Streamlit UI (Improved)
st.title("🎬 Movie Recommender System")
st.markdown("Select a movie and get similar recommendations!")

selected_movie = st.selectbox("Choose a movie", movies["title"].values)

if st.button("Recommend"):
    recommendations = get_recommendations(selected_movie)
    st.success("Recommended Movies:")
    for movie in recommendations:
        st.write(f"- {movie}")