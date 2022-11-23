import pandas as pd
from pre_processing import preProcessing
from rank_based import topNoMovies
from collaborative_filtering import get_recommendations
from surprise.reader import Reader
from surprise.dataset import Dataset
from surprise.model_selection import train_test_split
from surprise.prediction_algorithms.knns import KNNBasic

#def load_data():
path = r'src/recommender_system/ratings.csv'
with open(path, 'rb') as csv_file:
    # csv_file = "ratings.csv"
    ratings = pd.read_csv(csv_file)  # into a dataframe
# Call preprocessor to clean
cleaned_ratings = preProcessing(ratings)
#return ratings


# Rank-based recommendation systems provide recommendations based on the most popular items
# Take average of all the ratings provided to each movie and then rank them based on their average rating.

# Calculating average ratings
average_rating = cleaned_ratings.groupby('movieId').mean()['rating']
# Calculating the count of ratings
count_rating = cleaned_ratings.groupby('movieId').count()['rating']
# Making a dataframe with the count and average of ratings
final_rating = pd.DataFrame({'avg_rating':average_rating, 'rating_count':count_rating})


# # Recommending top 5 movies with 50 minimum interactions based on popularity
print(list(topNoMovies(final_rating, 5, 50)))
# #Recommending top 5 movies with 100 minimum interactions based on popularity
print(list(topNoMovies(final_rating, 5, 100)))

# Instantiating Reader scale with expected rating scale
reader = Reader(rating_scale=(0, 5))

# Loading the rating dataset
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

# Splitting the data into train and test dataset
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

sim_options = {'name': 'cosine',
               'user_based': True}
# Using the optimal similarity measure for user-user based collaborative filtering
# Creating an instance of KNNBasic with optimal hyperparameter values
similarity_algo_optimized_user = KNNBasic(sim_options=sim_options, k=40, min_k=6,verbose=False)

# Training the algorithm on the trainset
similarity_algo_optimized_user.fit(trainset)

userId=4
topmovies=5
# Predicted top 5 movies for userId=4 with similarity based recommendation system
recommendations = get_recommendations(ratings,topmovies,userId,similarity_algo_optimized_user)
# Predict top 5 movies for userId=4 with similarity based recommendation system
# print(recommendations)

recommendations_df = pd.DataFrame(recommendations)
filename = 'recommendations_userId_'+str(userId)+'_for_top_'+str(topmovies)+'_movies.csv'
print(filename)
recommendations_df.to_csv(filename, index=False)

