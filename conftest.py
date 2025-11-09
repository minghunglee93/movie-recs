"""
Pytest configuration and fixtures for Movie Recommender tests
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from fastapi.testclient import TestClient
from data_loader import MovieLensLoader


@pytest.fixture(scope="session", autouse=True)
def load_data():
    """Load data once before all tests"""

    print("\n" + "="*60)
    print("Loading test data...")
    print("="*60)

    loader = MovieLensLoader(dataset_size='100k')

    # Try to load processed data, otherwise load and process
    try:
        loader.load_processed_data()
        print("✓ Loaded processed data")
    except:
        print("Processing data (this may take a moment)...")
        loader.load_data()
        loader.create_user_item_matrix()
        loader.save_processed_data()
        print("✓ Data processed and saved")

    print("="*60 + "\n")

    return loader


@pytest.fixture(scope="session")
def test_client(load_data):
    """Create test client with loaded data"""
    from app import app

    # Create client and trigger startup
    with TestClient(app) as client:
        # Wait for startup to complete
        response = client.get("/health")
        if response.status_code != 200:
            raise Exception("App failed to start properly")

        yield client


@pytest.fixture(scope="session")
def test_data(load_data):
    """Provide test data"""
    return {
        'test_user_id': 1,
        'test_movie_id': 1,
        'ratings': load_data.ratings,
        'movies': load_data.movies
    }