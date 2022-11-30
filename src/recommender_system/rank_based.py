# Create a function to find the top n movies for a recommendation based on the average ratings of movies
# Add a threshold for a minimum number of interactions for a movie to be considered for recommendation.

def topNoMovies(data, n, min_interaction=100):
    
    #Finding movies with minimum number of interactions
    recommendations = data[data['rating_count'] >= min_interaction]
    
    #Sorting values w.r.t average rating 
    recommendations = recommendations.sort_values(by='avg_rating', ascending=False)
    
    return recommendations.index[:n]
