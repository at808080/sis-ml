# to use, hit http://localhost:5000/get_recommendations?user_id=48 or http://localhost:5000/get_recommendations?meal_id=1

from flask import Flask, request, jsonify
import pandas as pd
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from collaborative_filtering import (
    train_user_user_collaborative_filtering,
    train_item_item_collaborative_filtering,
    train_user_item_collaborative_filtering,
)
from content_based_filtering import create_similarity_matrix, get_top_similar_meals

app = Flask(__name__)

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

# Create similarity matrix for content-based filtering
cosine_sim = create_similarity_matrix(meals_df)

@app.route('/get_recommendations', methods=['GET'])
def get_recommendations():
    user_id = request.args.get('user_id', type=int)
    meal_id = request.args.get('meal_id', type=int)
    num_recommendations = 50

    if user_id is not None:
        user_user_recs = user_user_algo.get_neighbors(user_id, k=num_recommendations)
        item_item_recs = item_item_algo.get_neighbors(user_id, k=num_recommendations)
        return jsonify({
            "user_user_recommendations": user_user_recs,
            "item_item_recommendations": item_item_recs
        })

    if meal_id is not None:
        similar_meals = get_top_similar_meals(meal_id, cosine_sim, meals_df, n=num_recommendations)
        return jsonify({
            "similar_meals": similar_meals.to_dict(orient='records')
        })

    return jsonify({"message": "Invalid request"}), 400

if __name__ == '__main__':
    app.run(debug=True)