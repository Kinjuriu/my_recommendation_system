import pandas as pd
import os
from pre_processing import pre_process_data
from utils import get_top_movies
from recommender import get_recommendations
from surprise.reader import Reader
from surprise.dataset import Dataset
from surprise.model_selection import train_test_split
from surprise.prediction_algorithms.knns import KNNBasic
from dotenv import load_dotenv

load_dotenv()
FILE_PATH = os.getenv('FILE_PATH')

def read_csv(filepath: str) -> pd.DataFrame:
    """
    Read a CSV file and return a dataframe.
    
    Args:
        filepath (str): Path to the CSV file
        
    Returns:
        DataFrame: DataFrame containing the data from the CSV file
    """
    with open(path, 'rb') as csv_file:
        return pd.read_csv(csv_file)



# You can then use this function to read a CSV file and store it in a dataframe like this:
path = FILE_PATH
ratings = read_csv(path)

# Call preprocessor to clean
cleaned_ratings = pre_process_data(ratings)
#return ratings

# Rank-based recommendation systems provide recommendations based on the most popular items
# Take average of all the ratings provided to each movie and then rank them based on their average rating.

# Calculate the average ratings
avg_rating = cleaned_ratings.groupby('movieId')['rating'].mean()

# Calculate the count of ratings
rating_count = cleaned_ratings.groupby('movieId')['rating'].count()

# Create a dataframe with the average ratings and rating counts
final_rating = pd.DataFrame({'avg_rating': avg_rating, 'rating_count': rating_count})

if __name__ == "__main__":
    # Recommend top 5 movies with 50 minimum interactions based on popularity
    print(list(get_top_movies(final_rating, 5, 50)))

    # Recommend top 5 movies with 100 minimum interactions based on popularity
    print(list(get_top_movies(final_rating, 5, 100)))


    # Set the rating scale for the reader
    reader = Reader(rating_scale=(0, 5))

    # Load the ratings data
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

    # Split the data into train and test sets
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

    # Set the options for the similarity measure
    sim_options = {'name': 'cosine', 'user_based': True}

    # Use the optimal similarity measure for user-user collaborative filtering
    # Create an instance of KNNBasic with the optimal hyperparameters
    similarity_algo_optimized_user = KNNBasic(
        sim_options=sim_options, k=40, min_k=6, verbose=False
    )

    # Train the algorithm on the trainset
    similarity_algo_optimized_user.fit(trainset)


    # Set the user ID and number of movies to recommend
    user_id = 4
    top_movies = 5

    # Get recommendations using the similarity-based recommendation system
    recommendations = get_recommendations(
        ratings, top_movies, user_id, similarity_algo_optimized_user
    )

    # Convert recommendations to a dataframe
    recommendations_df = pd.DataFrame(recommendations)

    # Create a filename using the user ID and number of movies
    filename = f'recommendations_userId_{user_id}_for_top_{top_movies}_movies.csv'

    # Save the recommendations to a CSV file
    recommendations_df.to_csv(filename, index=False)
