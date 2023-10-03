import pandas as pd
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from collaborative_filtering import (
    train_user_user_collaborative_filtering,
    train_item_item_collaborative_filtering,
    train_user_item_collaborative_filtering,
    user_item_collaborative_filtering,
)
from content_based_filtering import create_similarity_matrix, get_top_similar_meals

# Load and preprocess data for collaborative filtering
ratings_df = pd.read_csv('data/ratings.csv')
# Load and preprocess data for content-based filtering
meals_df = pd.read_csv('data/meals.csv')
users_df = pd.read_csv('data/users.csv')

# Surprise Reader object specifies rating scale
reader = Reader(rating_scale=(1, 5))

# Load the data into the Surprise Dataset format
data = Dataset.load_from_df(ratings_df[['user_id', 'meal_id', 'rating']], reader)

# Split data into train, test sets
trainset, testset = train_test_split(data, test_size=0.25)

# Train collaborative filtering models
user_user_algo = train_user_user_collaborative_filtering(trainset)
item_item_algo = train_item_item_collaborative_filtering(trainset)
user_item_algo_cf = train_user_item_collaborative_filtering(trainset)

# Perform collaborative filtering recommendations
user_id = 48
num_recommendations = 5

user_user_recs = user_user_algo.get_neighbors(user_id, k=num_recommendations)
item_item_recs = item_item_algo.get_neighbors(user_id, k=num_recommendations)
# user_item_recs_cf = user_item_collaborative_filtering(user_id, num_recommendations, user_item_algo_cf, ratings_df, meals_df)

# Create similarity matrix for content-based filtering
cosine_sim = create_similarity_matrix(meals_df)

# Example: Get top 5 similar meals to a given meal_id (change as needed)
meal_id = 1  
similar_meals = get_top_similar_meals(meal_id, cosine_sim, meals_df, n=5)

# Print or use the collaborative filtering and content-based filtering results
print("\n")
print("Collaborative Filtering - User-User Recommendations:", user_user_recs)
print("\n")
print("Collaborative Filtering - Item-Item Recommendations:", item_item_recs)
# print("Collaborative Filtering - User-Item Recommendations:", user_item_recs_cf)
print("\n")
print("\nContent-Based Filtering - Top 5 Similar Meals to Meal ID", meal_id)
print(similar_meals[['meal_id', 'meal_name', 'category', 'cuisine', 'temperature']])