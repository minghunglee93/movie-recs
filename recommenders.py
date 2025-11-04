"""
Movie Recommendation Algorithms
Implements multiple recommendation strategies
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from typing import List, Tuple, Dict
import pickle
from pathlib import Path


class CollaborativeFilteringRecommender:
    """User-based and Item-based Collaborative Filtering"""
    
    def __init__(self, ratings_df, movies_df):
        """
        Initialize recommender
        
        Args:
            ratings_df: DataFrame with userId, movieId, rating
            movies_df: DataFrame with movieId, title, genres
        """
        self.ratings = ratings_df
        self.movies = movies_df
        
        # Create user-item matrix
        self.user_item_matrix = ratings_df.pivot(
            index='userId',
            columns='movieId',
            values='rating'
        ).fillna(0)
        
        # Compute similarity matrices
        self.user_similarity = None
        self.item_similarity = None
        
    def compute_user_similarity(self):
        """Compute user-user similarity matrix"""
        print("Computing user similarity matrix...")
        
        # Normalize by user mean
        user_mean = self.user_item_matrix.mean(axis=1)
        matrix_normalized = self.user_item_matrix.sub(user_mean, axis=0).fillna(0)
        
        # Compute cosine similarity
        self.user_similarity = cosine_similarity(matrix_normalized)
        
        print(f"✓ User similarity matrix computed: {self.user_similarity.shape}")
    
    def compute_item_similarity(self):
        """Compute item-item similarity matrix"""
        print("Computing item similarity matrix...")
        
        # Transpose to get item-based
        self.item_similarity = cosine_similarity(self.user_item_matrix.T)
        
        print(f"✓ Item similarity matrix computed: {self.item_similarity.shape}")
    
    def recommend_user_based(self, user_id: int, n: int = 10, 
                            min_similarity: float = 0.3) -> pd.DataFrame:
        """
        User-based collaborative filtering recommendations
        
        Args:
            user_id: User to get recommendations for
            n: Number of recommendations
            min_similarity: Minimum similarity threshold
            
        Returns:
            DataFrame with recommended movies
        """
        if self.user_similarity is None:
            self.compute_user_similarity()
        
        # Get user index
        try:
            user_idx = self.user_item_matrix.index.get_loc(user_id)
        except KeyError:
            print(f"User {user_id} not found")
            return pd.DataFrame()
        
        # Get similar users
        user_sims = self.user_similarity[user_idx]
        similar_users = np.argsort(user_sims)[::-1][1:]  # Exclude self
        
        # Filter by minimum similarity
        similar_users = [u for u in similar_users 
                        if user_sims[u] >= min_similarity][:50]
        
        if not similar_users:
            print("No similar users found")
            return pd.DataFrame()
        
        # Get movies user hasn't rated
        user_rated = set(self.user_item_matrix.iloc[user_idx][
            self.user_item_matrix.iloc[user_idx] > 0
        ].index)
        
        # Predict ratings for unrated movies
        predictions = {}
        
        for movie_id in self.user_item_matrix.columns:
            if movie_id in user_rated:
                continue
            
            # Weighted average of similar users' ratings
            numerator = 0
            denominator = 0
            
            for similar_user_idx in similar_users:
                rating = self.user_item_matrix.iloc[similar_user_idx][movie_id]
                if rating > 0:
                    sim = user_sims[similar_user_idx]
                    numerator += sim * rating
                    denominator += sim
            
            if denominator > 0:
                predictions[movie_id] = numerator / denominator
        
        # Sort and get top N
        top_movies = sorted(predictions.items(), 
                          key=lambda x: x[1], 
                          reverse=True)[:n]
        
        # Create results DataFrame
        results = pd.DataFrame(top_movies, columns=['movieId', 'predicted_rating'])
        results = results.merge(self.movies, on='movieId')
        
        return results[['movieId', 'title', 'genres', 'predicted_rating']]
    
    def recommend_item_based(self, user_id: int, n: int = 10) -> pd.DataFrame:
        """
        Item-based collaborative filtering recommendations
        
        Args:
            user_id: User to get recommendations for
            n: Number of recommendations
            
        Returns:
            DataFrame with recommended movies
        """
        if self.item_similarity is None:
            self.compute_item_similarity()
        
        # Get user index
        try:
            user_idx = self.user_item_matrix.index.get_loc(user_id)
        except KeyError:
            print(f"User {user_id} not found")
            return pd.DataFrame()
        
        # Get movies user has rated
        user_ratings = self.user_item_matrix.iloc[user_idx]
        rated_movies = user_ratings[user_ratings > 0]
        
        if len(rated_movies) == 0:
            print("User has no ratings")
            return pd.DataFrame()
        
        # Predict ratings for all unrated movies
        predictions = {}
        
        for movie_id in self.user_item_matrix.columns:
            if user_ratings[movie_id] > 0:
                continue  # Already rated
            
            # Get movie index
            movie_idx = self.user_item_matrix.columns.get_loc(movie_id)
            
            # Weighted sum of similar items
            numerator = 0
            denominator = 0
            
            for rated_movie_id, rating in rated_movies.items():
                rated_movie_idx = self.user_item_matrix.columns.get_loc(rated_movie_id)
                sim = self.item_similarity[movie_idx][rated_movie_idx]
                
                if sim > 0:
                    numerator += sim * rating
                    denominator += sim
            
            if denominator > 0:
                predictions[movie_id] = numerator / denominator
        
        # Sort and get top N
        top_movies = sorted(predictions.items(), 
                          key=lambda x: x[1], 
                          reverse=True)[:n]
        
        # Create results DataFrame
        results = pd.DataFrame(top_movies, columns=['movieId', 'predicted_rating'])
        results = results.merge(self.movies, on='movieId')
        
        return results[['movieId', 'title', 'genres', 'predicted_rating']]
    
    def get_similar_items(self, movie_id: int, n: int = 10) -> pd.DataFrame:
        """
        Find similar movies
        
        Args:
            movie_id: Movie to find similar movies for
            n: Number of similar movies
            
        Returns:
            DataFrame with similar movies
        """
        if self.item_similarity is None:
            self.compute_item_similarity()
        
        try:
            movie_idx = self.user_item_matrix.columns.get_loc(movie_id)
        except KeyError:
            print(f"Movie {movie_id} not found")
            return pd.DataFrame()
        
        # Get similarities
        sims = self.item_similarity[movie_idx]
        similar_idx = np.argsort(sims)[::-1][1:n+1]  # Exclude self
        
        similar_movies = []
        for idx in similar_idx:
            similar_movie_id = self.user_item_matrix.columns[idx]
            similar_movies.append({
                'movieId': similar_movie_id,
                'similarity': sims[idx]
            })
        
        results = pd.DataFrame(similar_movies)
        results = results.merge(self.movies, on='movieId')
        
        return results[['movieId', 'title', 'genres', 'similarity']]


class MatrixFactorizationRecommender:
    """Matrix Factorization using SVD"""
    
    def __init__(self, ratings_df, movies_df, n_factors=50):
        """
        Initialize recommender
        
        Args:
            ratings_df: DataFrame with userId, movieId, rating
            movies_df: DataFrame with movieId, title, genres
            n_factors: Number of latent factors
        """
        self.ratings = ratings_df
        self.movies = movies_df
        self.n_factors = n_factors
        
        # Create user-item matrix
        self.user_item_matrix = ratings_df.pivot(
            index='userId',
            columns='movieId',
            values='rating'
        ).fillna(0)
        
        self.svd_model = None
        self.user_factors = None
        self.item_factors = None
        
    def train(self):
        """Train SVD model"""
        print(f"Training SVD with {self.n_factors} factors...")
        
        # Apply SVD
        self.svd_model = TruncatedSVD(n_components=self.n_factors, random_state=42)
        self.user_factors = self.svd_model.fit_transform(self.user_item_matrix)
        self.item_factors = self.svd_model.components_.T
        
        # Explained variance
        explained_var = self.svd_model.explained_variance_ratio_.sum()
        print(f"✓ Model trained. Explained variance: {explained_var:.2%}")
    
    def predict_rating(self, user_id: int, movie_id: int) -> float:
        """Predict rating for a user-movie pair"""
        
        if self.svd_model is None:
            self.train()
        
        try:
            user_idx = self.user_item_matrix.index.get_loc(user_id)
            movie_idx = self.user_item_matrix.columns.get_loc(movie_id)
        except KeyError:
            return 0.0
        
        # Dot product of user and item factors
        prediction = np.dot(self.user_factors[user_idx], 
                          self.item_factors[movie_idx])
        
        # Clip to valid rating range
        return np.clip(prediction, 0.5, 5.0)
    
    def recommend(self, user_id: int, n: int = 10) -> pd.DataFrame:
        """
        Get recommendations for a user
        
        Args:
            user_id: User to get recommendations for
            n: Number of recommendations
            
        Returns:
            DataFrame with recommended movies
        """
        if self.svd_model is None:
            self.train()
        
        try:
            user_idx = self.user_item_matrix.index.get_loc(user_id)
        except KeyError:
            print(f"User {user_id} not found")
            return pd.DataFrame()
        
        # Get movies user hasn't rated
        user_ratings = self.user_item_matrix.iloc[user_idx]
        unrated_movies = user_ratings[user_ratings == 0].index
        
        # Predict ratings
        predictions = []
        for movie_id in unrated_movies:
            pred_rating = self.predict_rating(user_id, movie_id)
            predictions.append({
                'movieId': movie_id,
                'predicted_rating': pred_rating
            })
        
        # Sort and get top N
        results = pd.DataFrame(predictions)
        results = results.sort_values('predicted_rating', ascending=False).head(n)
        results = results.merge(self.movies, on='movieId')
        
        return results[['movieId', 'title', 'genres', 'predicted_rating']]


class PopularityRecommender:
    """Simple popularity-based recommender"""
    
    def __init__(self, ratings_df, movies_df):
        """Initialize recommender"""
        self.ratings = ratings_df
        self.movies = movies_df
        
        # Compute popularity metrics
        self.popularity = ratings_df.groupby('movieId').agg({
            'rating': ['count', 'mean']
        }).reset_index()
        self.popularity.columns = ['movieId', 'num_ratings', 'avg_rating']
        
        # Weighted rating (Bayesian average)
        C = self.popularity['avg_rating'].mean()
        m = self.popularity['num_ratings'].quantile(0.7)
        
        def weighted_rating(x):
            v = x['num_ratings']
            R = x['avg_rating']
            return (v/(v+m) * R) + (m/(m+v) * C)
        
        self.popularity['score'] = self.popularity.apply(weighted_rating, axis=1)
        self.popularity = self.popularity.merge(movies, on='movieId')
    
    def recommend(self, n: int = 10) -> pd.DataFrame:
        """Get top N popular movies"""
        return self.popularity.sort_values('score', ascending=False).head(n)[
            ['movieId', 'title', 'genres', 'num_ratings', 'avg_rating', 'score']
        ]
    
    def recommend_by_genre(self, genre: str, n: int = 10) -> pd.DataFrame:
        """Get popular movies in a specific genre"""
        genre_movies = self.popularity[
            self.popularity['genres'].str.contains(genre, case=False, na=False)
        ]
        return genre_movies.sort_values('score', ascending=False).head(n)[
            ['movieId', 'title', 'genres', 'num_ratings', 'avg_rating', 'score']
        ]


# Example usage
if __name__ == "__main__":
    from data_loader import MovieLensLoader
    
    # Load data
    loader = MovieLensLoader(dataset_size='100k')
    ratings, movies = loader.load_data()
    
    print("\n" + "="*60)
    print("TESTING RECOMMENDATION ALGORITHMS")
    print("="*60)
    
    # Test user ID
    test_user_id = 1
    
    # 1. Collaborative Filtering
    print("\n1. Collaborative Filtering")
    print("-" * 60)
    cf = CollaborativeFilteringRecommender(ratings, movies)
    
    print("\nUser-based recommendations:")
    user_recs = cf.recommend_user_based(test_user_id, n=5)
    print(user_recs)
    
    print("\nItem-based recommendations:")
    item_recs = cf.recommend_item_based(test_user_id, n=5)
    print(item_recs)
    
    # 2. Matrix Factorization
    print("\n2. Matrix Factorization (SVD)")
    print("-" * 60)
    mf = MatrixFactorizationRecommender(ratings, movies, n_factors=50)
    svd_recs = mf.recommend(test_user_id, n=5)
    print(svd_recs)
    
    # 3. Popularity-based
    print("\n3. Popularity-based")
    print("-" * 60)
    pop = PopularityRecommender(ratings, movies)
    print("\nTop popular movies:")
    print(pop.recommend(n=5))
    
    print("\nPopular Action movies:")
    print(pop.recommend_by_genre('Action', n=5))
    
    print("\n✓ All algorithms tested successfully!")
