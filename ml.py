import pandas as pd
from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import train_test_split
from surprise import accuracy
from surprise import SVD

# Load sample data from CSV files
meals_df = pd.read_csv('data/meals.csv')
users_df = pd.read_csv('data/users.csv')
ratings_df = pd.read_csv('data/ratings.csv')


###
### COLLABORATIVE FILTERING
###

# Surprise Reader object specifies rating scale
reader = Reader(rating_scale=(1, 5))

# Load the data into the Surprise Dataset format
data = Dataset.load_from_df(ratings_df[['user_id', 'meal_id', 'rating']], reader)

# Split data into train, test sets
trainset, testset = train_test_split(data, test_size=0.25)

# Train user-user collaborative filtering model - in this case KNNBasic)
user_user_algo = KNNBasic(sim_options={'user_based': True})
user_user_algo.fit(trainset)

# Train item-item collaborative filtering model - in this case KNNBasic
item_item_algo = KNNBasic(sim_options={'user_based': False})
item_item_algo.fit(trainset)


# Train a user-item collaborative filtering model - in this case SVD
user_item_algo = SVD()
user_item_algo.fit(trainset) 



num_meals = len(meals_df)
num_users = len(users_df)
num_ratings = len(ratings_df)

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





# User ID and number of recommendations
user_id = 48
num_recommendations = 5

#print("hyb ", user_id, users_df.loc[user_id, 'user_name'], num_recommendations)
user_user_recs = user_user_algo.get_neighbors(user_id, k=num_recommendations)

# Item-Item Collaborative Filtering Recommendations
item_item_recs = item_item_algo.get_neighbors(user_id, k=num_recommendations)

# User-Item Collaborative Filtering Recommendations
user_item_recs = user_item_collaborative_filtering(user_id, num_recommendations)

# Filter the ratings_df DataFrame to get reviewed meals for the user
user_reviews = ratings_df[ratings_df['user_id'] == user_id]

# Get the list of meal IDs reviewed by the user
reviewed_meals = user_reviews['meal_id'].tolist()


#Print original meals reviewed by the user and their ratings
for meal_id in reviewed_meals:
    meal_name = meals_df[(meals_df['meal_id'] == meal_id)]["meal_name"].values[0] # meals_df.loc[meal_id, 'meal_name']
    
    user_rating = ratings_df[(ratings_df['user_id'] == user_id) & (ratings_df['meal_id'] == meal_id)]['rating'].values[0]
    
    print(f"Meal: {meal_name} (Meal ID: {meal_id}), Rating: {user_rating}")

# Print User-User Collaborative Filtering Recommendations
print("\nUser-User Collaborative Filtering Recommendations:", user_user_recs)

# Print Item-Item Collaborative Filtering Recommendations
print("\nItem-Item Collaborative Filtering Recommendations:", item_item_recs)

# Print User-Item Collaborative Filtering Recommendations
print("\nUser-Item Collaborative Filtering Recommendations:", user_item_recs)




###
### CONTENT BASED FILTERING
###

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load data
meals_df = pd.read_csv('meals.csv')
users_df = pd.read_csv('users.csv')
ratings_df = pd.read_csv('ratings.csv')

# Merge user ratings with meal information
user_meal_ratings = pd.merge(ratings_df, meals_df, on='meal_id')

# Select relevant columns for content-based filtering
content_cols = ['meal_id', 'meal_name', 'category', 'calories', 'protein', 'fat', 'carbohydrates', 'cuisine', 'temperature']

# Fill missing values in categorical columns
meals_df['cuisine'].fillna('Unknown', inplace=True)
meals_df['temperature'].fillna('Unknown', inplace=True)

# Convert categorical columns to text
for col in ['category', 'cuisine', 'temperature']:
    meals_df[col] = meals_df[col].astype(str)

# Create a bag of words (BoW) representation for text-based columns
vectorizer = CountVectorizer()
text_features = vectorizer.fit_transform(meals_df['category'] + ' ' + meals_df['cuisine'] + ' ' + meals_df['temperature'])

# Calculate cosine similarity for text-based features
cosine_sim = cosine_similarity(text_features, text_features)

# Helper function to get top N similar meals
def get_top_similar_meals(meal_id, cosine_sim, n=10):
    sim_scores = list(enumerate(cosine_sim[meal_id]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:n+1]
    meal_indices = [i[0] for i in sim_scores]
    return meal_indices

# Example: Get top 5 similar meals to a given meal_id (change as needed)
meal_id = 1  
similar_meal_indices = get_top_similar_meals(meal_id, cosine_sim, n=5)

similar_meals = meals_df.iloc[similar_meal_indices]
print("Top 5 Similar Meals to Meal ID", meal_id)
print(similar_meals[['meal_id', 'meal_name', 'category', 'cuisine', 'temperature']])




