import os
from dotenv import load_dotenv

# Get the root folder path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Path to .env
ENV_PATH = os.path.join(PROJECT_ROOT, '.env')

# Load environment variables
load_dotenv(dotenv_path=ENV_PATH)

# TMDB API Key
TMDB_API_KEY = os.getenv('TMDB_API_KEY')

# Data folders
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
POSTER_DIR = os.path.join(DATA_DIR, 'posters')