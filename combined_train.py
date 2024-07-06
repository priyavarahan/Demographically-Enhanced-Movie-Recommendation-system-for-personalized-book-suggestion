import os
import pandas as pd
import joblib
from surprise import SVD, Reader, Dataset

# Set the path to the directory containing the model files
model_dir = '/opt/ml/model/'

# Load the movie model
movie_model_filename = os.path.join(model_dir, 'movie_model.joblib')
movie_algo = joblib.load(movie_model_filename)

# Load the book model
book_model_filename = os.path.join(model_dir, 'book_model.joblib')
book_algo = joblib.load(book_model_filename)

# Load the label encoder for genres
label_encoder_filename = os.path.join(model_dir, 'label_encoder.joblib')
label_encoder = joblib.load(label_encoder_filename)

# Load the book dataset
book_df = pd.read_csv('s3://dataset-bucket-project/book_train_data.csv')

# Function to get book recommendations based on movie genres
def get_book_recommendations(movie_genres, N=1):
    # Use LabelEncoder to convert string values in the 'genre' column to numerical values
    book_df['genre'] = label_encoder.transform(book_df['genre'])

    # Assuming book_df is the DataFrame containing book information
    matching_books = book_df[book_df['genre'].apply(lambda book_genre: book_genre in movie_genres)]

    if len(matching_books) == 0:
        return []

    recommended_books = matching_books.sample(min(N, len(matching_books)))['title'].tolist()

    return recommended_books

def predict_rating(user_id, movie_id):
    # Assuming movie_df is the DataFrame containing movie information
    movie_df = pd.read_csv('s3://dataset-bucket-project/movie_train_data.csv')
    
    # Get the rating prediction for the specified user and movie
    prediction = movie_algo.predict(user_id, movie_id)

    return prediction.est

# Example usage for testing
if __name__ == '__main__':
    # Example for getting book recommendations based on movie genres
    movie_genres = ['Action', 'Adventure']
    recommended_books = get_book_recommendations(movie_genres, N=3)
    print("Recommended Books:", recommended_books)

    # Example for predicting a rating
    user_id = '123'
    movie_id = '456'
    rating_prediction = predict_rating(user_id, movie_id)
    print("Predicted Rating:", rating_prediction)