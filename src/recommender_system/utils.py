# Create a function to find the top n movies for a recommendation based on the average ratings of movies
# Add a threshold for a minimum number of interactions for a movie to be considered for recommendation.

def get_top_movies(data, n, min_interactions=100):
    """
    Get the top number of movies with at least min_interactions ratings.
    
    Args:
        data (DataFrame): DataFrame containing movie ratings data
        n (int): Number of top movies to return
        min_interactions (int): Minimum number of ratings a movie must have
        
    Returns:
        Index: Index of top n movies with at least min_interactions ratings
    """
    
    # Get movies with at least min_interactions ratings
    recommendations = data[data['rating_count'] >= min_interactions]
    
    # Sort movies by average rating in descending order
    recommendations = recommendations.sort_values(by='avg_rating', ascending=False)
    
    # Return the index of the top n movies
    return recommendations.index[:n]

