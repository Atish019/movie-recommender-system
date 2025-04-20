import pandas as pd
import json

def extract_text_from_json(json_str):
    if pd.isna(json_str):
        return ""
    data = json.loads(json_str)
    return " ".join([x["name"] for x in data])

def preprocess_data():
    # Load data
    movies = pd.read_csv("data/raw/movies.csv")
    credits = pd.read_csv("data/raw/credits.csv")
    
    # Merge datasets
    movies = movies.merge(credits, on="title")
    
    # Extract genres & keywords from JSON
    movies["genres_str"] = movies["genres"].apply(extract_text_from_json)
    movies["keywords_str"] = movies["keywords"].apply(extract_text_from_json)
    
    # Combine features
    movies["combined_features"] = (
        movies["overview"].fillna("") + " " +
        movies["genres_str"] + " " +
        movies["keywords_str"]
    )
    
    # Save processed data
    movies.to_csv("data/processed/movies_processed.csv", index=False)

if __name__ == "__main__":
    preprocess_data()