# third parties imports
from pathlib import Path


class Constant:

    DATA_PATH = Path('data/small')  # -- fill here the dataset size to use

    # Content
    CONTENT_PATH = DATA_PATH / 'content'
    # - item
    ITEMS_FILENAME = 'movies.csv'
    ITEMS_FILENAME2 = 'movies2.csv'
    ITEM_ID_COL = 'movieId'
    LABEL_COL = 'title'
    GENRES_COL = 'genres'
    GENRE_FILENAME = 'movies.csv'

    # Evidence
    EVIDENCE_PATH = DATA_PATH / 'evidence'
    # - ratings
    RATINGS_FILENAME = 'ratings.csv'
    USER_ID_COL = 'userId'
    RATING_COL = 'rating'
    TIMESTAMP_COL = 'timestamp'
    USER_ITEM_RATINGS = [USER_ID_COL, ITEM_ID_COL, RATING_COL]

    # Rating scale
    RATINGS_SCALE = (1,5)  # -- fill in here the ratings scale as a tuple (min_value, max_value)

    # Evaluation
    EVALUATION_PATH = DATA_PATH / 'evaluations'

    # Recommendations
    REC_PATH = DATA_PATH / 'recs'


    # Posters
    POSTERS_PATH = "posters"