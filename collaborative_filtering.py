from surprise import KNNBasic, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

def train_user_user_collaborative_filtering(trainset):
    user_user_algo = KNNBasic(sim_options={'user_based': True})
    user_user_algo.fit(trainset)
    return user_user_algo

def train_item_item_collaborative_filtering(trainset):
    item_item_algo = KNNBasic(sim_options={'user_based': False})
    item_item_algo.fit(trainset)
    return item_item_algo

def train_user_item_collaborative_filtering(trainset):
    user_item_algo = SVD()
    user_item_algo.fit(trainset)
    return user_item_algo

# User-Item Collaborative Filtering Recommendations
def user_item_collaborative_filtering(user_id, num_recommendations=5):
    # Get list of all meal IDs
    all_meal_ids = [meal_id for meal_id in range(1, num_meals + 1)]
    print("all_meal_ids", all_meal_ids)
    # Remove meal IDs that the user has already rated
    user_rated_meals = ratings_df[ratings_df['user_id'] == user_id]['meal_id'].tolist()
    remaining_meal_ids = list(set(all_meal_ids) - set(user_rated_meals))
    print("user_rated_meals", user_rated_meals)
    print("remaining_meal_ids", remaining_meal_ids)
    # Predict ratings for remaining meals
    predictions = [(meal_id, user_item_algo.predict(user_id, meal_id).est) for meal_id in remaining_meal_ids]
    print("predictions", predictions)
    # Sort predictions by estimated rating in descending order
    sorted_predictions = sorted(predictions, key=lambda x: x[1], reverse=True)
    print("sorted_predictions", sorted_predictions)
    # Get the top N recommendations
    top_recommendations = [meal_id for meal_id, _ in sorted_predictions[:num_recommendations]]
    print("top_recommendations", top_recommendations)
    return top_recommendations
