from surprise import KNNBasic
import pandas as pd
from sklearn.model_selection import GridSearchCV
from surprise import accuracy

def train_item_based_model(trainset, testset, sim_options={'name': 'cosine', 'user_based': False}):
    """
    Trains an item-based collaborative filtering recommendation system using KNNBasic algorithm.
    
    Parameters
    ----------
    trainset: surprise.Trainset
        The trainset used to fit the KNNBasic algorithm.
    testset: list of tuples
        The testset used to predict the ratings.
    sim_options: dict, optional (default={'name': 'cosine', 'user_based': False})
        The similarity options for the KNNBasic algorithm. 
        Available options are 'cosine' and 'msd'.
    
    Returns
    -------
    predictions: list of surprise.Prediction objects
        The predictions made by the KNNBasic algorithm on the testset.
    """
    # Define the KNNBasic algorithm
    algo = KNNBasic(sim_options=sim_options, verbose=False)
    
    # Fit the KNNBasic algorithm on the trainset
    algo.fit(trainset)
    
    # Predict the ratings on the testset
    predictions = algo.test(testset)
    
    return predictions

def tune_hyperparameters(data, param_grid={'k': [20, 30,40], 'min_k': [3,6,9],
              'sim_options': {'name': ['msd', 'cosine'],
                              'user_based': [False]}
              }, cv=3):
    """
    Performs a hyperparameter tuning on a KNNBasic algorithm using grid search cross-validation.
    
    Parameters
    ----------
    data: surprise.Dataset
        The dataset used to perform the grid search cross-validation.
    param_grid: dict, optional (default={'k': [20, 30,40], 'min_k': [3,6,9],
              'sim_options': {'name': ['msd', 'cosine'],
                              'user_based': [False]}
              })
        The parameter grid to search through.
    cv: int, optional (default=3)
        The number of folds to use for cross-validation.
    
    Returns
    -------
    best_score: float
        The best root mean squared error (RMSE) score found through the grid search cross-validation.
    best_params: dict
        The combination of parameters that gave the best RMSE score.
    results_df: pandas.DataFrame
        The dataframe containing the results from the grid search cross-validation.
    """
    # Perform grid search cross-validation
    grid_obj = GridSearchCV(KNNBasic, param_grid, measures=['rmse', 'mae'], cv=cv, n_jobs=-1)
    grid_obj.fit(data)
    
    # Get the best RMSE score and best parameters
    best_score = grid_obj.best_score['rmse']
    best_params = grid_obj.best_params['rmse']

def train_recommendation_model(trainset, testset, param_grid, measures=['rmse', 'mae'], cv=3, n_jobs=-1):
    """
    Train recommendation model using grid search cross-validation
    
    Parameters:
    trainset (Dataset): trainset for the recommendation system
    testset (Dataset): testset for the recommendation system
    param_grid (dict): Hyperparameters to tune
    measures (list): Evaluation metrics, default ['rmse', 'mae']
    cv (int): Number of folds for cross-validation, default 3
    n_jobs (int): Number of parallel jobs, default -1
    
    Returns:
    best_model (object): Optimized recommendation model
    results_df (pd.DataFrame): Results of the grid search cross-validation
    best_params (dict): Optimal hyperparameters
    """
    grid_obj = GridSearchCV(KNNBasic, param_grid, measures=measures, cv=cv, n_jobs=n_jobs)
    grid_obj.fit(trainset)
    
    best_model = KNNBasic(sim_options={'name': grid_obj.best_params['rmse']['sim_options']['name'], 
                                       'user_based': grid_obj.best_params['rmse']['sim_options']['user_based']}, 
                         k=grid_obj.best_params['rmse']['k'],
                         min_k=grid_obj.best_params['rmse']['min_k'],
                         verbose=False)
    best_model.fit(trainset)
    
    results_df = pd.DataFrame.from_dict(grid_obj.cv_results)
    best_params = grid_obj.best_params['rmse']
    
    return best_model, results_df, best_params

def evaluate_recommendation_model(recommendation_model, testset, accuracy):
    """
    Evaluate recommendation model using Root Mean Squared Error (RMSE)
    
    Parameters:
    recommendation_model (object): Trained recommendation model
    testset (Dataset): testset for the recommendation system
    accuracy (module): Accuracy module from surprise
    
    Returns:
    rmse (float): RMSE of the recommendation model
    """
    predictions = recommendation_model.test(testset)
    rmse = accuracy.rmse(predictions)
    
    return rmse

