"""
Test suite for Movie Recommendation System
"""

import pytest
from data_loader import MovieLensLoader
from recommenders import (
    CollaborativeFilteringRecommender,
    MatrixFactorizationRecommender,
    PopularityRecommender
)


class TestDataLoader:
    """Test data loading functionality"""

    def test_data_download(self, load_data):
        """Test data can be downloaded"""
        assert (load_data.data_dir / 'ratings.csv').exists()
        assert (load_data.data_dir / 'movies.csv').exists()

    def test_load_data(self, load_data):
        """Test data loading"""
        ratings = load_data.ratings
        movies = load_data.movies

        assert not ratings.empty
        assert not movies.empty
        assert 'userId' in ratings.columns
        assert 'movieId' in ratings.columns
        assert 'rating' in ratings.columns
        assert 'title' in movies.columns

    def test_statistics(self, load_data):
        """Test statistics computation"""
        stats = load_data.get_statistics()

        assert 'num_users' in stats
        assert 'num_movies' in stats
        assert 'num_ratings' in stats
        assert stats['num_users'] > 0
        assert stats['num_movies'] > 0

    def test_popular_movies(self, load_data):
        """Test popular movies retrieval"""
        popular = load_data.get_popular_movies(n=10)

        assert len(popular) == 10
        assert 'title' in popular.columns
        assert 'num_ratings' in popular.columns


class TestRecommenders:
    """Test recommendation algorithms"""

    def test_collaborative_filtering_init(self, test_data):
        """Test CF initialization"""
        ratings = test_data['ratings']
        movies = test_data['movies']
        cf = CollaborativeFilteringRecommender(ratings, movies)

        assert cf.user_item_matrix is not None
        assert cf.user_item_matrix.shape[0] > 0

    def test_collaborative_filtering_recommendations(self, test_data):
        """Test CF recommendations"""
        ratings = test_data['ratings']
        movies = test_data['movies']
        test_user_id = test_data['test_user_id']

        cf = CollaborativeFilteringRecommender(ratings, movies)

        # Item-based recommendations
        recs = cf.recommend_item_based(test_user_id, n=5)

        assert not recs.empty
        assert len(recs) <= 5
        assert 'title' in recs.columns
        assert 'predicted_rating' in recs.columns

    def test_similar_movies(self, test_data):
        """Test similar movie finding"""
        ratings = test_data['ratings']
        movies = test_data['movies']
        test_movie_id = test_data['test_movie_id']

        cf = CollaborativeFilteringRecommender(ratings, movies)
        cf.compute_item_similarity()

        similar = cf.get_similar_items(test_movie_id, n=5)

        assert not similar.empty
        assert len(similar) <= 5
        assert 'similarity' in similar.columns

    def test_matrix_factorization_init(self, test_data):
        """Test MF initialization"""
        ratings = test_data['ratings']
        movies = test_data['movies']
        mf = MatrixFactorizationRecommender(ratings, movies, n_factors=20)

        assert mf.n_factors == 20
        assert mf.user_item_matrix is not None

    def test_matrix_factorization_train(self, test_data):
        """Test MF training"""
        ratings = test_data['ratings']
        movies = test_data['movies']
        mf = MatrixFactorizationRecommender(ratings, movies, n_factors=20)
        mf.train()

        assert mf.user_factors is not None
        assert mf.item_factors is not None
        assert mf.user_factors.shape[1] == 20

    def test_matrix_factorization_recommendations(self, test_data):
        """Test MF recommendations"""
        ratings = test_data['ratings']
        movies = test_data['movies']
        test_user_id = test_data['test_user_id']

        mf = MatrixFactorizationRecommender(ratings, movies, n_factors=20)

        recs = mf.recommend(test_user_id, n=5)

        assert not recs.empty
        assert len(recs) <= 5
        assert 'predicted_rating' in recs.columns

    def test_popularity_recommender(self, test_data):
        """Test popularity recommender"""
        ratings = test_data['ratings']
        movies = test_data['movies']
        pop = PopularityRecommender(ratings, movies)

        recs = pop.recommend(n=10)

        assert not recs.empty
        assert len(recs) == 10
        assert 'score' in recs.columns
        assert 'num_ratings' in recs.columns

    def test_popularity_by_genre(self, test_data):
        """Test genre-based popularity"""
        ratings = test_data['ratings']
        movies = test_data['movies']
        pop = PopularityRecommender(ratings, movies)

        recs = pop.recommend_by_genre('Action', n=5)

        assert not recs.empty
        assert all('Action' in g for g in recs['genres'])


class TestAPI:
    """Test API endpoints"""

    def test_root_endpoint(self, test_client):
        """Test root endpoint"""
        response = test_client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert 'message' in data
        assert 'algorithms' in data

    def test_health_check(self, test_client):
        """Test health check"""
        response = test_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'healthy'
        assert 'num_movies' in data

    def test_list_movies(self, test_client):
        """Test movie listing"""
        response = test_client.get("/movies?limit=10")
        assert response.status_code == 200
        data = response.json()
        assert len(data) <= 10
        assert 'movieId' in data[0]
        assert 'title' in data[0]

    def test_search_movies(self, test_client):
        """Test movie search"""
        response = test_client.get("/movies?search=toy&limit=5")
        assert response.status_code == 200
        data = response.json()
        assert len(data) > 0
        # Check that results contain 'toy'
        assert any('toy' in movie['title'].lower() for movie in data)

    def test_filter_by_genre(self, test_client):
        """Test genre filtering"""
        response = test_client.get("/movies?genre=Comedy&limit=5")
        assert response.status_code == 200
        data = response.json()
        assert all('Comedy' in movie['genres'] for movie in data)

    def test_get_movie(self, test_client, test_data):
        """Test get movie details"""
        test_movie_id = test_data['test_movie_id']
        response = test_client.get(f"/movies/{test_movie_id}")
        assert response.status_code == 200
        data = response.json()
        assert data['movieId'] == test_movie_id
        assert 'title' in data

    def test_get_movie_not_found(self, test_client):
        """Test get non-existent movie"""
        response = test_client.get("/movies/999999")
        assert response.status_code == 404

    def test_popular_movies(self, test_client):
        """Test popular movies endpoint"""
        response = test_client.get("/movies/popular?n=5")
        assert response.status_code == 200
        data = response.json()
        assert len(data['recommendations']) == 5
        assert data['algorithm'] == 'popularity'

    def test_similar_movies(self, test_client, test_data):
        """Test similar movies endpoint"""
        test_movie_id = test_data['test_movie_id']
        response = test_client.get(f"/movies/{test_movie_id}/similar?n=5")
        assert response.status_code == 200
        data = response.json()
        assert 'similar_movies' in data
        assert len(data['similar_movies']) <= 5

    def test_get_recommendations(self, test_client, test_data):
        """Test recommendations endpoint"""
        test_user_id = test_data['test_user_id']
        response = test_client.get(
            f"/users/{test_user_id}/recommendations",
            params={"algorithm": "collaborative", "n": 5}
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data['recommendations']) <= 5
        assert data['algorithm'] == 'collaborative'

    def test_get_recommendations_different_algorithms(self, test_client, test_data):
        """Test all algorithms"""
        test_user_id = test_data['test_user_id']
        algorithms = ['collaborative', 'matrix_factorization', 'popularity']

        for algo in algorithms:
            response = test_client.get(
                f"/users/{test_user_id}/recommendations",
                params={"algorithm": algo, "n": 3}
            )
            assert response.status_code == 200
            data = response.json()
            assert data['algorithm'] == algo
            assert len(data['recommendations']) > 0

    def test_user_history(self, test_client, test_data):
        """Test user history endpoint"""
        test_user_id = test_data['test_user_id']
        response = test_client.get(f"/users/{test_user_id}/history?limit=10")
        assert response.status_code == 200
        data = response.json()
        assert 'history' in data
        assert 'total_ratings' in data
        assert 'avg_rating' in data

    def test_rate_movie(self, test_client, test_data):
        """Test rating a movie"""
        test_user_id = test_data['test_user_id']
        test_movie_id = test_data['test_movie_id']

        response = test_client.post(
            f"/users/{test_user_id}/rate",
            json={
                "user_id": test_user_id,
                "movie_id": test_movie_id,
                "rating": 4.5
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'success'
        assert data['rating'] == 4.5

    def test_rate_invalid_movie(self, test_client, test_data):
        """Test rating non-existent movie"""
        test_user_id = test_data['test_user_id']

        response = test_client.post(
            f"/users/{test_user_id}/rate",
            json={
                "user_id": test_user_id,
                "movie_id": 999999,
                "rating": 4.5
            }
        )
        assert response.status_code == 404

    def test_get_statistics(self, test_client):
        """Test statistics endpoint"""
        response = test_client.get("/stats")
        assert response.status_code == 200
        data = response.json()
        assert 'statistics' in data
        assert 'num_users' in data['statistics']


class TestPerformance:
    """Test performance characteristics"""

    def test_recommendation_speed(self, test_client, test_data):
        """Test recommendation response time"""
        import time

        test_user_id = test_data['test_user_id']

        # Warm up - first request takes longer
        test_client.get(
            f"/users/{test_user_id}/recommendations",
            params={"algorithm": "popularity", "n": 5}
        )

        # Now test actual speed (should be faster after warm-up)
        start = time.time()
        response = test_client.get(
            f"/users/{test_user_id}/recommendations",
            params={"algorithm": "popularity", "n": 10}
        )
        end = time.time()

        assert response.status_code == 200
        # Popularity should be very fast
        assert (end - start) < 0.5  # Under 500ms

        # Test matrix factorization (should also be fast)
        start = time.time()
        response = test_client.get(
            f"/users/{test_user_id}/recommendations",
            params={"algorithm": "matrix_factorization", "n": 10}
        )
        end = time.time()

        assert response.status_code == 200
        # Matrix factorization should be reasonably fast
        assert (end - start) < 2.0  # Under 2 seconds

    def test_concurrent_requests(self, test_client, test_data):
        """Test handling multiple concurrent requests"""
        import concurrent.futures

        test_user_id = test_data['test_user_id']

        def make_request():
            return test_client.get(
                f"/users/{test_user_id}/recommendations",
                params={"algorithm": "popularity", "n": 5}
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            results = [f.result() for f in futures]

        assert all(r.status_code == 200 for r in results)


# Run tests with pytest
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])