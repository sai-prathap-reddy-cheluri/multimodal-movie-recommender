import os

import requests
from dotenv import load_dotenv

# Load the environment variables from .env file
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__),'..', '.env'))

TMDB_API_KEY = os.getenv('TMDB_API_KEY')

def fetch_popular_movies():
    """Fetch the top 5 popular movies from TMDB."""
    url = f'https://api.themoviedb.org/3/movie/popular'
    params = {
        'api_key': TMDB_API_KEY,
        'language': 'en-us',
        'page': 1
    }

    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        print("Top 5 popular movies:")
        for movie in data['results'][:5]:
            print(f"{movie['title']} ({movie['release_date']})")
    else:
        print("Failed to fetch movies:", response.status_code, response.text)

if __name__ == '__main__':
    fetch_popular_movies()