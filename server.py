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
from endpoint_helpers import getMealPlan
import json


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
        # item_item_recs = item_item_algo.get_neighbors(user_id, k=num_recommendations)

        user_user_recs_rich = []
        for meal_id_ in user_user_recs:
            print("meal id", meal_id_)
            meal_info = meals_df[meals_df['meal_id'] == meal_id_] #.iloc[0]  # using .iloc[0] to get the first row as a Series
            print("meal info", meal_info)
            #meal_dict = meal_info.to_dict()  # Convert the Series to a dictionary
            meal_dict = {}
            for column in meals_df.columns:
                meal_dict[column] = meal_info[column]
            
            print("meal dict", meal_dict)# json.dumps(meal_dict))


            # user_user_recs_rich.append(json.dumps(meal_dict))
            print("baking ", meal_info['baking'])
            # print(meal_info['baking'])
            user_user_recs_rich.append({
                "baking": meal_info['baking'],
                "calories": meal_info['calories'],
                "carbohydrates": meal_info['carbohydrates'],
                "category": meal_info['category'],
                "cuisine": meal_info['cuisine'],
                "fat": meal_info['fat'],
                "frying": meal_info['frying'],
                "gluten": meal_info['gluten'],
                "grilling": meal_info['grilling'],
                "lactose": meal_info['lactose'],
                "meal_id": meal_info['meal_id'],
                "meal_name": meal_info['meal_name'],
                "nuts": meal_info['nuts'],
                "preparation_time": meal_info['preparation_time'],
                "protein": meal_info['protein'],
                "roasting": meal_info['roasting'],
                "temperature": meal_info['temperature'],
                "vegetarian": meal_info['vegetarian']
            })

        return jsonify({
            "user_user_recommendations": user_user_recs
            # user_user_recs_rich,
            # "item_item_recommendations": item_item_recs
        })

    if meal_id is not None:
        similar_meals = get_top_similar_meals(meal_id, cosine_sim, meals_df, n=num_recommendations)
        return jsonify({
            "similar_meals": similar_meals.to_dict(orient='records')
        })

    return jsonify({"message": "Invalid request"}), 400



@app.route('/get_meal_plan', methods=['GET'])
def get_meal_plan():
    # Get parameters from the request's query string
    days = request.args.get('days', type=int)
    mealLikes = request.args.getlist('mealLikes', type=int)
    mealDislikes = request.args.getlist('mealDislikes', type=int)
    dietaryReqs = request.args.getlist('dietaryReqs', type=str)

    # Check if required parameters are provided
    if days is None or mealLikes is None or mealDislikes is None or dietaryReqs is None:
        return jsonify({"error": "Missing parameters"}), 400

    # Call the getMealPlan function with the provided parameters
    meal_plan = getMealPlan(days, mealLikes, mealDislikes, dietaryReqs)

    # Return the result as a JSON response
    # return jsonify({"meal_plan": meal_plan})

    days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]

    print(json.dumps({"meal_plan": meal_plan[0]}))

    return json.dumps({"meal_plan": meal_plan})
    


if __name__ == '__main__':
    app.run(debug=True)