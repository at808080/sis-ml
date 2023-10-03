from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def create_similarity_matrix(meals_df):
    # Create a bag of words (BoW) representation for text-based columns
    vectorizer = CountVectorizer()
    text_features = vectorizer.fit_transform(meals_df['category'] + ' ' + meals_df['cuisine'] + ' ' + meals_df['temperature'])

    # Calculate cosine similarity for text-based features
    cosine_sim = cosine_similarity(text_features, text_features)
    return cosine_sim

def get_top_similar_meals(meal_id, cosine_sim, meals_df, n=10):
    sim_scores = list(enumerate(cosine_sim[meal_id]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:n+1]
    meal_indices = [i[0] for i in sim_scores]
    return meals_df.iloc[meal_indices]