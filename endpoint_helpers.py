# to use, hit http://localhost:5000/get_recommendations?user_id=48 or http://localhost:5000/get_recommendations?meal_id=1

from flask import jsonify
import pandas as pd
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from collaborative_filtering import (
    train_user_user_collaborative_filtering,
    train_item_item_collaborative_filtering,
    train_user_item_collaborative_filtering,
)
from content_based_filtering import create_similarity_matrix, get_top_similar_meals, get_top_similar_mealss

import json

all_meals_df = pd.read_csv('data/meals.csv')

def getMeal(meal_id):
    meal = all_meals_df[all_meals_df['meal_id'] == meal_id]
    
    # Check if a meal with the given meal_id exists
    if not meal.empty:
        return meal
    else:
        return None  # Return None if the meal_id is not found

def getMealProp(meal_id, prop):
    return getMeal(meal_id)[prop].values[0]

# Surprise Reader object specifies rating scale
reader = Reader(rating_scale=(1, 5))

userId = 1

mealId = 2

numRecommendations = 50

def testRecsByMeal(meal_id, num_recommendations):
    return getRecommendationsByMeal(meal_id, num_recommendations, all_meals_df)

def getRecommendationsByMeal(meal_id, num_recommendations, mealsDf):
    cosine_sim = create_similarity_matrix(mealsDf)
    similar_meals = get_top_similar_meals(meal_id, cosine_sim, mealsDf, n=num_recommendations)
    return similar_meals

def getRecommendationsByMeall(meal_id, num_recommendations, mealsDf):
    mealsDfReindexed = mealsDf.reset_index()

    cosine_sim = create_similarity_matrix(mealsDfReindexed)
    similar_meals = get_top_similar_mealss(meal_id, cosine_sim, mealsDfReindexed, n=num_recommendations)
    
    return similar_meals

def getFilteredMealsDataframe(allMealsDf, mealDislikes, dietaryReqs, categories):
    # Remove entries with meal_ids in mealDislikes
    allMealsDf = allMealsDf[~allMealsDf['meal_id'].isin(mealDislikes)]
    
    # Remove entries with false values in the specified dietary columns
    for dietaryReq in dietaryReqs:
        if dietaryReq in allMealsDf.columns:
            allMealsDf = allMealsDf[allMealsDf[dietaryReq] != False]

    if len(categories) > 0:
        allMealsDf = allMealsDf[allMealsDf['category'].isin(categories)]
    
    return allMealsDf

def getMealPlan(days, mealLikes, mealDislikes, dietaryReqs):
    lunchdinnerfilteredMealsDf = getFilteredMealsDataframe(all_meals_df, mealDislikes, dietaryReqs, [])
    breakfastfilteredMealsDf = getFilteredMealsDataframe(all_meals_df, mealDislikes, dietaryReqs, ["Breakfast"])

    # mealLikeIdx = 0

    lunchdinnerlikes = []
    for mealId in mealLikes: 
        if getMealProp(mealId,"category") != "Breakfast": lunchdinnerlikes.append(mealId) 
    breakfastlikes = []
    for mealId in mealLikes: 
        if getMealProp(mealId,"category") == "Breakfast": breakfastlikes.append(mealId)

    # if len(lunchdinnerlikes) == 0:
    #     lunchdinnerlikes.append() 

    # lunchdinnermealsrequired = days * 2
    # breakfastmealsrequired = days    


    print(breakfastfilteredMealsDf)

    print("\n\n")

    breakfastRecs = getRecommendationsByMeall(breakfastlikes[-1], days, breakfastfilteredMealsDf)
    lunchDinnerRecs = getRecommendationsByMeall(lunchdinnerlikes[-1], days * 2, lunchdinnerfilteredMealsDf )

    # print("lunchdinnerrecs\n")
    # # print(lunchDinnerRecs)
    # print(lunchDinnerRecs.iloc[0])
    # print("\n\n")

    mealPlan = []
    for day in range(0, days):
        mealPlan.append({
            "breakfast": breakfastRecs.iloc[day]["meal_id"],
            "lunch": lunchDinnerRecs.iloc[day*2]["meal_id"],#.values[0],
            "dinner": lunchDinnerRecs.iloc[day*2+1]["meal_id"]#.values[0],
        })

    return mealPlan






