"""
Test suite for Movie Recommendation System
"""

import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from app import app
from data_loader import MovieLensLoader
from recommenders import (
    CollaborativeFilteringRecommender,
    MatrixFactorizationRecommender,
    PopularityRecommender
)

# Test client
client = TestClient(app)

# Test data
TEST_USER_ID = 1
TEST_MOVIE_ID = 1


class TestDataLoader:
    """Test data loading functionality"""
    
    @pytest.fixture
    def loader(self):
        return MovieLensLoader(dataset_size='100k')
    
    def test_data_download(self, loader):
        """Test data can be downloaded"""
        loader.download_data()
        assert (loader.data_dir / 'ratings.csv').exists()
        assert (loader.data_dir / 'movies.csv').exists()
    
    def test_load_data(self, loader):
        """Test data loading"""
        ratings, movies = loader.load_data()
        
        assert not ratings.empty
        assert not movies.empty
        assert 'userId' in ratings.columns
        assert 'movieId' in ratings.columns
        assert 'rating' in ratings.columns
        assert 'title' in movies.columns
    
    def test_statistics(self, loader):
        """Test statistics computation"""
        loader.load_data()
        stats = loader.get_statistics()
        
        assert 'num_users' in stats
        assert 'num_movies' in stats
        assert 'num_ratings' in stats
        assert stats['num_users'] > 0
        assert stats['num_movies'] > 0
    
    def test_popular_movies(self, loader):
        """Test popular movies retrieval"""
        loader.load_data()
        popular = loader.get_popular_movies(n=10)
        
        assert len(popular) == 10
        assert 'title' in popular.columns
        assert 'num_ratings' in popular.columns


class TestRecommenders:
    """Test recommendation algorithms"""
    
    @pytest.fixture
    def data(self):
        """Load data for testing"""
        loader = MovieLensLoader(dataset_size='100k')
        ratings, movies = loader.load_data()
        return ratings, movies
    
    def test_collaborative_filtering_init(self, data):
        """Test CF initialization"""
        ratings, movies = data
        cf = CollaborativeFilteringRecommender(ratings, movies)
        
        assert cf.user_item_matrix is not None
        assert cf.user_item_matrix.shape[0] > 0
    
    def test_collaborative_filtering_recommendations(self, data):
        """Test CF recommendations"""
        ratings, movies = data
        cf = CollaborativeFilteringRecommender(ratings, movies)
        
        # Item-based recommendations
        recs = cf.recommend_item_based(TEST_USER_ID, n=5)
        
        assert not recs.empty
        assert len(recs) <= 5
        assert 'title' in recs.columns
        assert 'predicted_rating' in recs.columns
    
    def test_similar_movies(self, data):
        """Test similar movie finding"""
        ratings, movies = data
        cf = CollaborativeFilteringRecommender(ratings, movies)
        cf.compute_item_similarity()
        
        similar = cf.get_similar_items(TEST_MOVIE_ID, n=5)
        
        assert not similar.empty
        assert len(similar) <= 5
        assert 'similarity' in similar.columns
    
    def test_matrix_factorization_init(self, data):
        """Test MF initialization"""
        ratings, movies = data
        mf = MatrixFactorizationRecommender(ratings, movies, n_factors=20)
        
        assert mf.n_factors == 20
        assert mf.user_item_matrix is not None
    
    def test_matrix_factorization_train(self, data):
        """Test MF training"""
        ratings, movies = data
        mf = MatrixFactorizationRecommender(ratings, movies, n_factors=20)
        mf.train()
        
        assert mf.user_factors is not None
        assert mf.item_factors is not None
        assert mf.user_factors.shape[1] == 20
    
    def test_matrix_factorization_recommendations(self, data):
        """Test MF recommendations"""
        ratings, movies = data
        mf = MatrixFactorizationRecommender(ratings, movies, n_factors=20)
        
        recs = mf.recommend(TEST_USER_ID, n=5)
        
        assert not recs.empty
        assert len(recs) <= 5
        assert 'predicted_rating' in recs.columns
    
    def test_popularity_recommender(self, data):
        """Test popularity recommender"""
        ratings, movies = data
        pop = PopularityRecommender(ratings, movies)
        
        recs = pop.recommend(n=10)
        
        assert not recs.empty
        assert len(recs) == 10
        assert 'score' in recs.columns
        assert 'num_ratings' in recs.columns
    
    def test_popularity_by_genre(self, data):
        """Test genre-based popularity"""
        ratings, movies = data
        pop = PopularityRecommender(ratings, movies)
        
        recs = pop.recommend_by_genre('Action', n=5)
        
        assert not recs.empty
        assert all('Action' in g for g in recs['genres'])


class TestAPI:
    """Test API endpoints"""
    
    def test_root_endpoint(self):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert 'message' in data
        assert 'algorithms' in data
    
    def test_health_check(self):
        """Test health check"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'healthy'
        assert 'num_movies' in data
    
    def test_list_movies(self):
        """Test movie listing"""
        response = client.get("/movies?limit=10")
        assert response.status_code == 200
        data = response.json()
        assert len(data) <= 10
        assert 'movieId' in data[0]
        assert 'title' in data[0]
    
    def test_search_movies(self):
        """Test movie search"""
        response = client.get("/movies?search=toy&limit=5")
        assert response.status_code == 200
        data = response.json()
        assert len(data) > 0
        # Check that results contain 'toy'
        assert any('toy' in movie['title'].lower() for movie in data)
    
    def test_filter_by_genre(self):
        """Test genre filtering"""
        response = client.get("/movies?genre=Comedy&limit=5")
        assert response.status_code == 200
        data = response.json()
        assert all('Comedy' in movie['genres'] for movie in data)
    
    def test_get_movie(self):
        """Test get movie details"""
        response = client.get(f"/movies/{TEST_MOVIE_ID}")
        assert response.status_code == 200
        data = response.json()
        assert data['movieId'] == TEST_MOVIE_ID
        assert 'title' in data
    
    def test_get_movie_not_found(self):
        """Test get non-existent movie"""
        response = client.get("/movies/999999")
        assert response.status_code == 404
    
    def test_popular_movies(self):
        """Test popular movies endpoint"""
        response = client.get("/movies/popular?n=5")
        assert response.status_code == 200
        data = response.json()
        assert len(data['recommendations']) == 5
        assert data['algorithm'] == 'popularity'
    
    def test_similar_movies(self):
        """Test similar movies endpoint"""
        response = client.get(f"/movies/{TEST_MOVIE_ID}/similar?n=5")
        assert response.status_code == 200
        data = response.json()
        assert 'similar_movies' in data
        assert len(data['similar_movies']) <= 5
    
    def test_get_recommendations(self):
        """Test recommendations endpoint"""
        response = client.get(
            f"/users/{TEST_USER_ID}/recommendations",
            params={"algorithm": "collaborative", "n": 5}
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data['recommendations']) <= 5
        assert data['algorithm'] == 'collaborative'
    
    def test_get_recommendations_different_algorithms(self):
        """Test all algorithms"""
        algorithms = ['collaborative', 'matrix_factorization', 'popularity']
        
        for algo in algorithms:
            response = client.get(
                f"/users/{TEST_USER_ID}/recommendations",
                params={"algorithm": algo, "n": 3}
            )
            assert response.status_code == 200
            data = response.json()
            assert data['algorithm'] == algo
            assert len(data['recommendations']) > 0
    
    def test_user_history(self):
        """Test user history endpoint"""
        response = client.get(f"/users/{TEST_USER_ID}/history?limit=10")
        assert response.status_code == 200
        data = response.json()
        assert 'history' in data
        assert 'total_ratings' in data
        assert 'avg_rating' in data
    
    def test_rate_movie(self):
        """Test rating a movie"""
        response = client.post(
            f"/users/{TEST_USER_ID}/rate",
            json={
                "user_id": TEST_USER_ID,
                "movie_id": TEST_MOVIE_ID,
                "rating": 4.5
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'success'
        assert data['rating'] == 4.5
    
    def test_rate_invalid_movie(self):
        """Test rating non-existent movie"""
        response = client.post(
            f"/users/{TEST_USER_ID}/rate",
            json={
                "user_id": TEST_USER_ID,
                "movie_id": 999999,
                "rating": 4.5
            }
        )
        assert response.status_code == 404
    
    def test_get_statistics(self):
        """Test statistics endpoint"""
        response = client.get("/stats")
        assert response.status_code == 200
        data = response.json()
        assert 'statistics' in data
        assert 'num_users' in data['statistics']


class TestPerformance:
    """Test performance characteristics"""
    
    def test_recommendation_speed(self):
        """Test recommendation response time"""
        import time
        
        start = time.time()
        response = client.get(
            f"/users/{TEST_USER_ID}/recommendations",
            params={"algorithm": "collaborative", "n": 10}
        )
        end = time.time()
        
        assert response.status_code == 200
        assert (end - start) < 1.0  # Should be under 1 second
    
    def test_concurrent_requests(self):
        """Test handling multiple concurrent requests"""
        import concurrent.futures
        
        def make_request():
            return client.get(
                f"/users/{TEST_USER_ID}/recommendations",
                params={"algorithm": "popularity", "n": 5}
            )
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            results = [f.result() for f in futures]
        
        assert all(r.status_code == 200 for r in results)


# Run tests with pytest
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
