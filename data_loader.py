"""
MovieLens Dataset Loader and Preprocessor
Downloads and prepares data for recommendation system
"""

import pandas as pd
import numpy as np
import requests
import zipfile
import os
from pathlib import Path
from typing import Tuple, Dict
import pickle

class MovieLensLoader:
    """Load and preprocess MovieLens dataset"""
    
    def __init__(self, data_dir='./data', dataset_size='100k'):
        """
        Initialize data loader
        
        Args:
            data_dir: Directory to store data
            dataset_size: '100k', '1m', or '25m'
        """
        self.data_dir = Path(data_dir)
        self.dataset_size = dataset_size
        self.data_dir.mkdir(exist_ok=True)
        
        # URLs for different dataset sizes
        self.urls = {
            '100k': 'https://files.grouplens.org/datasets/movielens/ml-latest-small.zip',
            '1m': 'https://files.grouplens.org/datasets/movielens/ml-1m.zip',
            '25m': 'https://files.grouplens.org/datasets/movielens/ml-25m.zip'
        }
        
        self.ratings = None
        self.movies = None
        self.user_item_matrix = None
        
    def download_data(self):
        """Download MovieLens dataset if not exists"""
        
        if self.dataset_size not in self.urls:
            raise ValueError(f"Dataset size must be one of: {list(self.urls.keys())}")
        
        url = self.urls[self.dataset_size]
        zip_path = self.data_dir / f'movielens-{self.dataset_size}.zip'
        
        # Check if already downloaded
        if (self.data_dir / 'ratings.csv').exists():
            print(f"✓ Dataset already exists in {self.data_dir}")
            return
        
        print(f"Downloading MovieLens {self.dataset_size} dataset...")
        print(f"URL: {url}")
        
        # Download
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(zip_path, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    print(f"\rProgress: {percent:.1f}%", end='')
        
        print("\n✓ Download complete!")
        
        # Extract
        print("Extracting files...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.data_dir)
        
        # Move files to data_dir root
        extracted_dir = list(self.data_dir.glob('ml-*'))[0]
        for file in extracted_dir.glob('*.csv'):
            target = self.data_dir / file.name
            if target.exists():
                target.unlink()  # Remove existing file
            file.rename(target)

        # Cleanup extracted directory and all remaining files
        import shutil
        shutil.rmtree(extracted_dir)
        zip_path.unlink()

        print(f"✓ Data extracted to {self.data_dir}")

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load ratings and movies data"""

        # Ensure data exists
        if not (self.data_dir / 'ratings.csv').exists():
            self.download_data()

        print("Loading data...")

        # Load ratings
        self.ratings = pd.read_csv(
            self.data_dir / 'ratings.csv',
            dtype={'userId': np.int32, 'movieId': np.int32,
                   'rating': np.float32, 'timestamp': np.int64}
        )

        # Load movies
        self.movies = pd.read_csv(
            self.data_dir / 'movies.csv',
            dtype={'movieId': np.int32}
        )

        print(f"✓ Loaded {len(self.ratings):,} ratings from {self.ratings.userId.nunique():,} users")
        print(f"✓ Loaded {len(self.movies):,} movies")

        return self.ratings, self.movies

    def create_user_item_matrix(self) -> pd.DataFrame:
        """Create user-item rating matrix"""

        if self.ratings is None:
            self.load_data()

        print("Creating user-item matrix...")

        # Pivot to create matrix
        self.user_item_matrix = self.ratings.pivot(
            index='userId',
            columns='movieId',
            values='rating'
        ).fillna(0)

        print(f"✓ Matrix shape: {self.user_item_matrix.shape}")
        print(f"  Sparsity: {(1 - np.count_nonzero(self.user_item_matrix) / self.user_item_matrix.size) * 100:.2f}%")

        return self.user_item_matrix

    def get_statistics(self) -> Dict:
        """Get dataset statistics"""

        if self.ratings is None:
            self.load_data()

        # Helper to convert numpy types to Python types
        def convert_to_python(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_python(v) for k, v in obj.items()}
            return obj

        stats = {
            'num_users': int(self.ratings.userId.nunique()),
            'num_movies': int(self.ratings.movieId.nunique()),
            'num_ratings': int(len(self.ratings)),
            'avg_rating': float(self.ratings.rating.mean()),
            'median_rating': float(self.ratings.rating.median()),
            'rating_distribution': {
                float(k): int(v) for k, v in
                self.ratings.rating.value_counts().sort_index().to_dict().items()
            },
            'sparsity': float(1 - (len(self.ratings) / (self.ratings.userId.nunique() * self.ratings.movieId.nunique()))),
            'ratings_per_user': convert_to_python(
                self.ratings.groupby('userId').size().describe().to_dict()
            ),
            'ratings_per_movie': convert_to_python(
                self.ratings.groupby('movieId').size().describe().to_dict()
            )
        }

        return stats

    def print_statistics(self):
        """Print dataset statistics"""

        stats = self.get_statistics()

        print("\n" + "="*60)
        print("MOVIELENS DATASET STATISTICS")
        print("="*60)
        print(f"Total Users:      {stats['num_users']:,}")
        print(f"Total Movies:     {stats['num_movies']:,}")
        print(f"Total Ratings:    {stats['num_ratings']:,}")
        print(f"Average Rating:   {stats['avg_rating']:.2f}")
        print(f"Median Rating:    {stats['median_rating']:.1f}")
        print(f"Sparsity:         {stats['sparsity']*100:.2f}%")
        print("\nRating Distribution:")
        for rating, count in sorted(stats['rating_distribution'].items()):
            print(f"  {rating:.1f} stars: {count:,} ({count/stats['num_ratings']*100:.1f}%)")
        print("\nRatings per User:")
        print(f"  Mean:   {stats['ratings_per_user']['mean']:.1f}")
        print(f"  Median: {stats['ratings_per_user']['50%']:.1f}")
        print(f"  Max:    {stats['ratings_per_user']['max']:.0f}")
        print("\nRatings per Movie:")
        print(f"  Mean:   {stats['ratings_per_movie']['mean']:.1f}")
        print(f"  Median: {stats['ratings_per_movie']['50%']:.1f}")
        print(f"  Max:    {stats['ratings_per_movie']['max']:.0f}")
        print("="*60)

    def get_popular_movies(self, n=10) -> pd.DataFrame:
        """Get most popular movies by number of ratings"""

        if self.ratings is None or self.movies is None:
            self.load_data()

        popular = (
            self.ratings.groupby('movieId')
            .agg({
                'rating': ['count', 'mean']
            })
            .reset_index()
        )
        popular.columns = ['movieId', 'num_ratings', 'avg_rating']
        popular = popular.merge(self.movies, on='movieId')
        popular = popular.sort_values('num_ratings', ascending=False).head(n)

        return popular

    def get_top_rated_movies(self, min_ratings=50, n=10) -> pd.DataFrame:
        """Get top rated movies with minimum number of ratings"""

        if self.ratings is None or self.movies is None:
            self.load_data()

        top_rated = (
            self.ratings.groupby('movieId')
            .agg({
                'rating': ['count', 'mean']
            })
            .reset_index()
        )
        top_rated.columns = ['movieId', 'num_ratings', 'avg_rating']
        top_rated = top_rated[top_rated.num_ratings >= min_ratings]
        top_rated = top_rated.merge(self.movies, on='movieId')
        top_rated = top_rated.sort_values('avg_rating', ascending=False).head(n)

        return top_rated

    def save_processed_data(self, output_path='./data/processed'):
        """Save processed data for faster loading"""

        output_path = Path(output_path)
        output_path.mkdir(exist_ok=True)

        if self.ratings is None:
            self.load_data()
        if self.user_item_matrix is None:
            self.create_user_item_matrix()

        # Save as pickle for faster loading
        with open(output_path / 'ratings.pkl', 'wb') as f:
            pickle.dump(self.ratings, f)
        with open(output_path / 'movies.pkl', 'wb') as f:
            pickle.dump(self.movies, f)
        with open(output_path / 'user_item_matrix.pkl', 'wb') as f:
            pickle.dump(self.user_item_matrix, f)

        print(f"✓ Saved processed data to {output_path}")

    def load_processed_data(self, input_path='./data/processed'):
        """Load pre-processed data"""

        input_path = Path(input_path)

        if not (input_path / 'ratings.pkl').exists():
            print("Processed data not found. Loading raw data...")
            self.load_data()
            self.create_user_item_matrix()
            return

        print("Loading processed data...")

        with open(input_path / 'ratings.pkl', 'rb') as f:
            self.ratings = pickle.load(f)
        with open(input_path / 'movies.pkl', 'rb') as f:
            self.movies = pickle.load(f)
        with open(input_path / 'user_item_matrix.pkl', 'rb') as f:
            self.user_item_matrix = pickle.load(f)

        print("✓ Loaded processed data")


# Example usage and testing
if __name__ == "__main__":
    # Initialize loader
    loader = MovieLensLoader(dataset_size='100k')

    # Load data
    ratings, movies = loader.load_data()

    # Print statistics
    loader.print_statistics()

    # Show popular movies
    print("\nMost Popular Movies:")
    print(loader.get_popular_movies(10)[['title', 'num_ratings', 'avg_rating']])

    # Show top rated movies
    print("\nTop Rated Movies (min 50 ratings):")
    print(loader.get_top_rated_movies(min_ratings=50, n=10)[['title', 'num_ratings', 'avg_rating']])

    # Create user-item matrix
    user_item_matrix = loader.create_user_item_matrix()

    # Save processed data
    loader.save_processed_data()

    print("\n✓ Data preparation complete!")