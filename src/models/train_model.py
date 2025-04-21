import mlflow
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def train_model():
    mlflow.start_run()
    
    # Load processed data
    movies = pd.read_csv("data/processed/movies_processed.csv")
    
    # TF-IDF Vectorization
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(movies["combined_features"])
    
    # Save TF-IDF model (for new predictions)
    with open("models/tfidf.pkl", "wb") as f:
        pickle.dump(tfidf, f)
    
    # Cosine Similarity
    cosine_sim = cosine_similarity(tfidf_matrix)
    
    # Save indices for movie titles (for faster lookup)
    movie_indices = pd.Series(movies.index, index=movies["title"]).to_dict()
    with open("models/movie_indices.pkl", "wb") as f:
        pickle.dump(movie_indices, f)
    
    # Log artifacts in MLflow
    mlflow.log_artifact("models/tfidf.pkl", "model")
    mlflow.log_artifact("models/movie_indices.pkl", "model")
    mlflow.end_run()

if __name__ == "__main__":
    train_model()