import pandas as pd
from surprise import SVD
from surprise import Dataset
from surprise import accuracy
from surprise.model_selection import GridSearchCV

def train_svd(trainset):
    """
    Train a SVD algorithm on a trainset.

    Parameters:
    trainset (surprise.Trainset): The trainset to train the SVD algorithm on.

    Returns:
    surprise.SVD: The trained SVD algorithm.
    """
    algo_svd = SVD()
    algo_svd.fit(trainset)
    return algo_svd

def predict_ratings(testset, algo_svd):
    """
    Predict ratings using a trained SVD algorithm on a testset.

    Parameters:
    testset (list of tuples): The testset to predict the ratings on.
    algo_svd (surprise.SVD): The trained SVD algorithm.

    Returns:
    list: List of predictions.
    """
    predictions = algo_svd.test(testset)
    return predictions

def compute_rmse(predictions):
    """
    Compute RMSE (Root Mean Squared Error) on predictions.

    Parameters:
    predictions (list): List of predictions.

    Returns:
    float: RMSE.
    """
    rmse = accuracy.rmse(predictions)
    return rmse

def grid_search_svd(data):
    """
    Perform hyperparameter tuning for a SVD algorithm using GridSearchCV.

    Parameters:
    data (surprise.Dataset): The dataset to perform hyperparameter tuning on.

    Returns:
    surprise.model_selection.GridSearchCV: The grid search object.
    """
    param_grid = {'n_epochs': [10, 20, 30], 'lr_all': [0.001, 0.005, 0.01], 'reg_all': [0.2, 0.4, 0.6]}
    gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3, n_jobs=-1)
    gs.fit(data)
    return gs

def get_best_score_rmse(gs):
    """
    Get the best RMSE score from a grid search object.

    Parameters:
    gs (surprise.model_selection.GridSearchCV): The grid search object.

    Returns:
    float: Best RMSE score.
    """
    best_score_rmse = gs.best_score['rmse']
    return best_score_rmse

def get_best_params_rmse(gs):
    """
    Get the combination of parameters that gave the best RMSE score from a grid search object.

    Parameters:
    gs (surprise.model_selection.GridSearchCV): The grid search object.

    Returns:
    dict: Combination of parameters that gave the best RMSE score.
    """
    best_params_rmse = gs.best_params['rmse']
    return best_params_rmse

def build_final_model(trainset, best_params):
    """
    Builds the final SVD model using the best hyperparameters from the grid search.
    
    Parameters:
        trainset (pandas dataframe): The training data for the SVD model
        best_params (dict): A dictionary of the best hyperparameters found during the grid search
        
    Returns:
        svd_algo_optimized (scikit-learn SVD model): The optimized SVD model
    """
    # Building the optimized SVD model using the best hyperparameters
    svd_algo_optimized = SVD(n_epochs=best_params['n_epochs'], 
                             lr_all=best_params['lr_all'], 
                             reg_all=best_params['reg_all'])
    
    # Training the algorithm on the trainset
    svd_algo_optimized.fit(trainset)
    
    return svd_algo_optimized

