# build a similarity-based recommendation systems using cosine similarity
# use KNN to find similar users which are the nearest neighbor to the given userid

# A performance metrics in surprise
from surprise import accuracy
# Class will be used to parse a file containing ratings, data should be in structure - user ; item ; rating
from surprise.reader import Reader
# Class for loading datasets
from surprise.dataset import Dataset
# For model tuning model hyper-parameters
from surprise.model_selection import GridSearchCV
# For splitting the rating data in train and test dataset
from surprise.model_selection import train_test_split
# For implementing similarity based recommendation system
from surprise.prediction_algorithms.knns import KNNBasic
# For implementing matrix factorization based recommendation system
from surprise.prediction_algorithms.matrix_factorization import SVD
# For implementing cross validation
from surprise.model_selection import KFold

def get_recommendations(data, user_id, top_n, algo):
    
    """
    Generates recommendations for a given user using the specified recommendation algorithm.
    
    Args:
        data (pandas.DataFrame): The data frame containing the user-item interactions.
        user_id (int): The user ID for which recommendations should be generated.
        top_n (int): The number of recommendations to generate.
        algo (obj): The recommendation algorithm to use.
        
    Returns:
        list: A list of tuples containing the recommended movie IDs and their predicted ratings.
    """

    # Creating an empty list to store the recommended movie ids
    recommendations = []
    
    # Creating an user item interactions matrix 
    user_item_interactions_matrix = data.pivot(index='userId', columns='movieId', values='rating')
    
    # Extracting those movie ids which the user_id has not interacted yet
    non_interacted_movies = user_item_interactions_matrix.loc[user_id][user_item_interactions_matrix.loc[user_id].isnull()].index.tolist()
    
    # Looping through each of the movie id which user_id has not interacted yet
    for item_id in non_interacted_movies:
        
        # Predicting the ratings for those non interacted movie ids by this user
        est = algo.predict(user_id, item_id).est
        
        # Appending the predicted ratings
        recommendations.append((item_id, est))

    # Sorting the predicted ratings in descending order
    recommendations.sort(key=lambda x: x[1], reverse=True)

    return recommendations[:top_n] # returing top n highest predicted rating movies for this user

def perform_tuning_and_rmse(data):
    """
    This function performs hyperparameter tuning for the KNN algorithm and
    returns the best rmse score and the combination of hyperparameters that gave the best score.
    """
    param_grid = {
        'k': [20, 30, 40],
        'min_k': [3, 6, 9],
        'sim_options': {
            'name': ['msd', 'cosine'],
            'user_based': [True]
        }
    }
    grid_obj = GridSearchCV(KNNBasic, param_grid, measures=['rmse', 'mae'], cv=3, n_jobs=-1)
    grid_obj.fit(data)
    best_rmse_score = grid_obj.best_score['rmse']
    best_params = grid_obj.best_params['rmse']
    return best_rmse_score, best_params

def build_final_model(trainset, testset, sim_options):
    """
    This function builds the final KNN model with optimal hyperparameters and returns
    the rmse score on the test set.
    """
    similarity_algo_optimized_user = KNNBasic(sim_options=sim_options, k=40, min_k=6, verbose=False)
    similarity_algo_optimized_user.fit(trainset)
    predictions = similarity_algo_optimized_user.test(testset)
    rmse = accuracy.rmse(predictions)
    return rmse