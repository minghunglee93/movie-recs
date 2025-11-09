#!/bin/bash

# Movie Recommender System - Quick Start Script

set -e

echo "=========================================="
echo "Movie Recommendation System Setup"
echo "=========================================="
echo

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# Check prerequisites
echo "Checking prerequisites..."

if command -v python3 &> /dev/null; then
    print_success "Python 3 found: $(python3 --version)"
else
    print_error "Python 3 not found. Please install Python 3.10+"
    exit 1
fi

echo

# Main menu
echo "What would you like to do?"
echo "1) Full setup (download data + install + run)"
echo "2) Just download data"
echo "3) Run API locally"
echo "4) Run with Docker"
echo "5) Run tests"
echo "6) Interactive demo"
echo
read -p "Enter choice [1-6]: " choice

case $choice in
    1)
        echo
        echo "=========================================="
        echo "Full Setup"
        echo "=========================================="
        
        # Create virtual environment
        if [ ! -d "venv" ]; then
            print_warning "Creating virtual environment..."
            python3 -m venv venv
            print_success "Virtual environment created"
        fi
        
        # Activate virtual environment
        source venv/bin/activate
        print_success "Virtual environment activated"
        
        # Install dependencies
        print_warning "Installing dependencies..."
        pip install -q --upgrade pip
        pip install -q -r requirements.txt
        print_success "Dependencies installed"
        
        # Download data
        print_warning "Downloading MovieLens dataset (this may take a minute)..."
        python data_loader.py
        print_success "Data downloaded and processed"
        
        # Run API
        print_success "Setup complete!"
        echo
        echo "Starting API server..."
        echo "Visit http://localhost:8000/docs for interactive API documentation"
        echo "Press Ctrl+C to stop"
        echo
        python app.py
        ;;
        
    2)
        echo
        echo "=========================================="
        echo "Downloading Dataset"
        echo "=========================================="
        
        if [ ! -d "venv" ]; then
            python3 -m venv venv
        fi
        
        source venv/bin/activate
        pip install -q pandas numpy requests
        
        print_warning "Downloading MovieLens 100k dataset..."
        python data_loader.py
        
        print_success "Dataset downloaded to ./data/"
        echo
        echo "Dataset includes:"
        echo "  - $(wc -l < data/ratings.csv) ratings"
        echo "  - $(wc -l < data/movies.csv) movies"
        ;;
        
    3)
        echo
        echo "=========================================="
        echo "Running API Locally"
        echo "=========================================="
        
        if [ ! -d "venv" ]; then
            print_error "Virtual environment not found. Run option 1 first."
            exit 1
        fi
        
        if [ ! -f "data/ratings.csv" ]; then
            print_error "Dataset not found. Run option 2 first."
            exit 1
        fi
        
        source venv/bin/activate
        
        print_success "Starting API server..."
        echo
        echo "API will be available at:"
        echo "  - Main: http://localhost:8000"
        echo "  - Docs: http://localhost:8000/docs"
        echo "  - Health: http://localhost:8000/health"
        echo
        echo "Press Ctrl+C to stop"
        echo
        
        python app.py
        ;;
        
    4)
        echo
        echo "=========================================="
        echo "Running with Docker"
        echo "=========================================="
        
        if ! command -v docker &> /dev/null; then
            print_error "Docker not found. Please install Docker first."
            exit 1
        fi
        
        # Build image
        print_warning "Building Docker image..."
        docker build -t movie-recommender:latest .
        print_success "Image built"
        
        # Stop existing container
        docker stop movie-recommender 2>/dev/null || true
        docker rm movie-recommender 2>/dev/null || true
        
        # Run container
        print_warning "Starting container..."
        docker run -d \
            --name movie-recommender \
            -p 8000:8000 \
            -v $(pwd)/data:/app/data \
            movie-recommender:latest
        
        print_success "Container started!"
        
        echo
        echo "Waiting for API to be ready..."
        sleep 15
        
        if curl -s http://localhost:8000/health | grep -q healthy; then
            print_success "API is healthy!"
            echo
            echo "Access at:"
            echo "  - API docs: http://localhost:8000/docs"
            echo "  - Health: http://localhost:8000/health"
            echo
            echo "View logs: docker logs -f movie-recommender"
            echo "Stop: docker stop movie-recommender"
        else
            print_error "API not responding. Check logs: docker logs movie-recommender"
        fi
        ;;
        
    5)
        echo
        echo "=========================================="
        echo "Running Tests"
        echo "=========================================="
        
        if [ ! -d "venv" ]; then
            print_error "Virtual environment not found. Run option 1 first."
            exit 1
        fi
        
        source venv/bin/activate
        pip install -q pytest pytest-asyncio
        
        print_warning "Running test suite..."
        echo
        
        pytest test_recommender.py -v
        
        if [ $? -eq 0 ]; then
            echo
            print_success "All tests passed!"
        else
            echo
            print_error "Some tests failed"
        fi
        ;;
        
    6)
        echo
        echo "=========================================="
        echo "Interactive Demo"
        echo "=========================================="
        
        if [ ! -d "venv" ]; then
            python3 -m venv venv
            source venv/bin/activate
            pip install -q -r requirements.txt
        else
            source venv/bin/activate
        fi
        
        # Start API in background
        print_warning "Starting API..."
        python app.py &
        API_PID=$!
        sleep 10
        
        print_success "API started!"
        echo
        
        # Interactive demo
        while true; do
            echo
            echo "Demo Options:"
            echo "1) Get recommendations for User 1"
            echo "2) Find movies similar to Toy Story"
            echo "3) Get popular movies"
            echo "4) Search for a movie"
            echo "5) Exit"
            echo
            read -p "Choose option: " demo_choice
            
            case $demo_choice in
                1)
                    echo
                    echo "Recommendations for User 1 (Collaborative Filtering):"
                    curl -s "http://localhost:8000/users/1/recommendations?algorithm=collaborative&n=5" | python -m json.tool
                    ;;
                2)
                    echo
                    echo "Movies similar to Toy Story (movieId=1):"
                    curl -s "http://localhost:8000/movies/1/similar?n=5" | python -m json.tool
                    ;;
                3)
                    echo
                    echo "Top 10 Popular Movies:"
                    curl -s "http://localhost:8000/movies/popular?n=10" | python -m json.tool
                    ;;
                4)
                    echo
                    read -p "Enter movie title to search: " search_term
                    curl -s "http://localhost:8000/movies?search=$search_term&limit=5" | python -m json.tool
                    ;;
                5)
                    echo
                    print_warning "Stopping API..."
                    kill $API_PID 2>/dev/null || true
                    print_success "Demo ended"
                    exit 0
                    ;;
                *)
                    print_error "Invalid option"
                    ;;
            esac
        done
        ;;
        
    *)
        print_error "Invalid choice"
        exit 1
        ;;
esac
