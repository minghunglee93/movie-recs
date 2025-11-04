"""
Movie Recommendation API
FastAPI application for serving recommendations
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import pandas as pd
from datetime import datetime
import logging

from data_loader import MovieLensLoader
from recommenders import (
    CollaborativeFilteringRecommender,
    MatrixFactorizationRecommender,
    PopularityRecommender
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Movie Recommendation API",
    description="Get personalized movie recommendations using multiple algorithms",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for recommenders
loader = None
cf_recommender = None
mf_recommender = None
pop_recommender = None
ratings_df = None
movies_df = None


# Response models
class MovieRecommendation(BaseModel):
    movieId: int
    title: str
    genres: str
    predicted_rating: Optional[float] = Field(None, ge=0, le=5)
    similarity: Optional[float] = Field(None, ge=0, le=1)
    score: Optional[float] = None

class RecommendationResponse(BaseModel):
    user_id: Optional[int] = None
    algorithm: str
    recommendations: List[MovieRecommendation]
    timestamp: str

class RatingRequest(BaseModel):
    user_id: int
    movie_id: int
    rating: float = Field(..., ge=0.5, le=5.0)

class Movie(BaseModel):
    movieId: int
    title: str
    genres: str


@app.on_event("startup")
async def startup_event():
    """Load data and initialize recommenders on startup"""
    global loader, cf_recommender, mf_recommender, pop_recommender
    global ratings_df, movies_df
    
    logger.info("Loading MovieLens data...")
    
    try:
        # Load data
        loader = MovieLensLoader(dataset_size='100k')
        
        # Try to load processed data first
        try:
            loader.load_processed_data()
        except:
            # If not available, load and process
            loader.load_data()
            loader.create_user_item_matrix()
            loader.save_processed_data()
        
        ratings_df = loader.ratings
        movies_df = loader.movies
        
        # Initialize recommenders
        logger.info("Initializing recommenders...")
        
        cf_recommender = CollaborativeFilteringRecommender(ratings_df, movies_df)
        cf_recommender.compute_item_similarity()  # Pre-compute for faster recommendations
        
        mf_recommender = MatrixFactorizationRecommender(ratings_df, movies_df, n_factors=50)
        mf_recommender.train()
        
        pop_recommender = PopularityRecommender(ratings_df, movies_df)
        
        logger.info("✓ All recommenders initialized successfully!")
        
    except Exception as e:
        logger.error(f"Failed to initialize recommenders: {e}")
        raise


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Movie Recommendation API",
        "version": "1.0.0",
        "algorithms": {
            "collaborative_filtering": "User-based and item-based collaborative filtering",
            "matrix_factorization": "SVD-based matrix factorization",
            "popularity": "Popularity-based recommendations"
        },
        "endpoints": {
            "GET /health": "Health check",
            "GET /movies": "List all movies",
            "GET /movies/popular": "Get popular movies",
            "GET /movies/{movie_id}": "Get movie details",
            "GET /movies/{movie_id}/similar": "Find similar movies",
            "GET /users/{user_id}/recommendations": "Get personalized recommendations",
            "GET /users/{user_id}/history": "Get user's rating history",
            "POST /users/{user_id}/rate": "Add a new rating"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "recommenders_loaded": all([
            cf_recommender is not None,
            mf_recommender is not None,
            pop_recommender is not None
        ]),
        "num_movies": len(movies_df) if movies_df is not None else 0,
        "num_users": ratings_df.userId.nunique() if ratings_df is not None else 0,
        "num_ratings": len(ratings_df) if ratings_df is not None else 0,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/movies", response_model=List[Movie])
async def list_movies(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    genre: Optional[str] = None,
    search: Optional[str] = None
):
    """
    List movies with optional filtering
    
    - **limit**: Number of movies to return
    - **offset**: Offset for pagination
    - **genre**: Filter by genre
    - **search**: Search in movie titles
    """
    movies = movies_df.copy()
    
    # Apply filters
    if genre:
        movies = movies[movies['genres'].str.contains(genre, case=False, na=False)]
    
    if search:
        movies = movies[movies['title'].str.contains(search, case=False, na=False)]
    
    # Pagination
    movies = movies.iloc[offset:offset+limit]
    
    return movies.to_dict('records')


@app.get("/movies/popular")
async def get_popular_movies(
    n: int = Query(10, ge=1, le=50),
    genre: Optional[str] = None
):
    """
    Get popular movies
    
    - **n**: Number of movies to return
    - **genre**: Optional genre filter
    """
    if genre:
        result = pop_recommender.recommend_by_genre(genre, n=n)
    else:
        result = pop_recommender.recommend(n=n)
    
    recommendations = result.to_dict('records')
    
    return RecommendationResponse(
        algorithm="popularity",
        recommendations=recommendations,
        timestamp=datetime.now().isoformat()
    )


@app.get("/movies/{movie_id}")
async def get_movie(movie_id: int):
    """Get details for a specific movie"""
    movie = movies_df[movies_df.movieId == movie_id]
    
    if movie.empty:
        raise HTTPException(status_code=404, detail="Movie not found")
    
    movie_data = movie.iloc[0].to_dict()
    
    # Add statistics
    movie_ratings = ratings_df[ratings_df.movieId == movie_id]
    
    if not movie_ratings.empty:
        movie_data['num_ratings'] = len(movie_ratings)
        movie_data['avg_rating'] = float(movie_ratings.rating.mean())
        movie_data['rating_distribution'] = movie_ratings.rating.value_counts().to_dict()
    
    return movie_data


@app.get("/movies/{movie_id}/similar")
async def get_similar_movies(
    movie_id: int,
    n: int = Query(10, ge=1, le=50)
):
    """
    Find similar movies based on collaborative filtering
    
    - **movie_id**: Movie to find similar movies for
    - **n**: Number of similar movies to return
    """
    result = cf_recommender.get_similar_items(movie_id, n=n)
    
    if result.empty:
        raise HTTPException(status_code=404, detail="Movie not found or no similar movies")
    
    recommendations = result.to_dict('records')
    
    return {
        "movie_id": movie_id,
        "algorithm": "item_similarity",
        "similar_movies": recommendations,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/users/{user_id}/recommendations")
async def get_recommendations(
    user_id: int,
    algorithm: str = Query("collaborative", regex="^(collaborative|matrix_factorization|popularity)$"),
    n: int = Query(10, ge=1, le=50)
):
    """
    Get personalized movie recommendations
    
    - **user_id**: User to get recommendations for
    - **algorithm**: Algorithm to use (collaborative, matrix_factorization, popularity)
    - **n**: Number of recommendations
    """
    
    # Check if user exists
    if user_id not in ratings_df.userId.values and algorithm != "popularity":
        raise HTTPException(
            status_code=404, 
            detail=f"User {user_id} not found. Try algorithm='popularity' for general recommendations."
        )
    
    try:
        if algorithm == "collaborative":
            result = cf_recommender.recommend_item_based(user_id, n=n)
        elif algorithm == "matrix_factorization":
            result = mf_recommender.recommend(user_id, n=n)
        elif algorithm == "popularity":
            result = pop_recommender.recommend(n=n)
        
        if result.empty:
            raise HTTPException(
                status_code=404,
                detail="No recommendations available for this user"
            )
        
        recommendations = result.to_dict('records')
        
        return RecommendationResponse(
            user_id=user_id if algorithm != "popularity" else None,
            algorithm=algorithm,
            recommendations=recommendations,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/users/{user_id}/history")
async def get_user_history(
    user_id: int,
    limit: int = Query(50, ge=1, le=500),
    sort_by: str = Query("timestamp", regex="^(timestamp|rating)$"),
    order: str = Query("desc", regex="^(asc|desc)$")
):
    """
    Get user's rating history
    
    - **user_id**: User ID
    - **limit**: Number of ratings to return
    - **sort_by**: Sort by timestamp or rating
    - **order**: Sort order (asc or desc)
    """
    user_ratings = ratings_df[ratings_df.userId == user_id].copy()
    
    if user_ratings.empty:
        raise HTTPException(status_code=404, detail="User not found or has no ratings")
    
    # Merge with movie info
    user_ratings = user_ratings.merge(movies_df, on='movieId')
    
    # Sort
    ascending = (order == "asc")
    user_ratings = user_ratings.sort_values(sort_by, ascending=ascending)
    
    # Limit
    user_ratings = user_ratings.head(limit)
    
    # Format timestamp
    user_ratings['timestamp'] = pd.to_datetime(user_ratings['timestamp'], unit='s')
    
    return {
        "user_id": user_id,
        "total_ratings": len(ratings_df[ratings_df.userId == user_id]),
        "avg_rating": float(ratings_df[ratings_df.userId == user_id].rating.mean()),
        "history": user_ratings[['movieId', 'title', 'genres', 'rating', 'timestamp']].to_dict('records')
    }


@app.post("/users/{user_id}/rate")
async def rate_movie(user_id: int, rating_request: RatingRequest):
    """
    Add a new rating (Note: In-memory only, not persisted)
    
    - **user_id**: User ID
    - **rating_request**: Rating details (movie_id, rating)
    """
    global ratings_df
    
    # Validate movie exists
    if rating_request.movie_id not in movies_df.movieId.values:
        raise HTTPException(status_code=404, detail="Movie not found")
    
    # Check if rating already exists
    existing = ratings_df[
        (ratings_df.userId == user_id) & 
        (ratings_df.movieId == rating_request.movie_id)
    ]
    
    if not existing.empty:
        # Update existing rating
        ratings_df.loc[
            (ratings_df.userId == user_id) & 
            (ratings_df.movieId == rating_request.movie_id),
            'rating'
        ] = rating_request.rating
        action = "updated"
    else:
        # Add new rating
        new_rating = pd.DataFrame([{
            'userId': user_id,
            'movieId': rating_request.movie_id,
            'rating': rating_request.rating,
            'timestamp': int(datetime.now().timestamp())
        }])
        ratings_df = pd.concat([ratings_df, new_rating], ignore_index=True)
        action = "added"
    
    logger.info(f"Rating {action}: User {user_id}, Movie {rating_request.movie_id}, Rating {rating_request.rating}")
    
    return {
        "status": "success",
        "action": action,
        "user_id": user_id,
        "movie_id": rating_request.movie_id,
        "rating": rating_request.rating,
        "message": f"Rating {action} successfully. Note: Changes are in-memory only."
    }


@app.get("/stats")
async def get_statistics():
    """Get dataset statistics"""
    stats = loader.get_statistics()
    
    return {
        "dataset": "MovieLens",
        "statistics": stats,
        "timestamp": datetime.now().isoformat()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
