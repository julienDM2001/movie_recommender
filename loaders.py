# third parties imports
import pandas as pd
from surprise import Reader
from surprise import Dataset
# local imports
from constants import Constant as C
import os
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

def load_ratings(surprise_format=False):
    df_ratings = pd.read_csv(C.EVIDENCE_PATH / C.RATINGS_FILENAME)
    if surprise_format == True:
        reader = Reader(rating_scale =(C.RATINGS_SCALE))
        df = Dataset.load_from_df(df_ratings[["userId", "movieId", "rating"]],reader)
        return df
    else:
        return df_ratings


def load_items():
    df_items = pd.read_csv(C.CONTENT_PATH / C.ITEMS_FILENAME)
    df_items = df_items.set_index(C.ITEM_ID_COL)
    return df_items


def export_evaluation_report(df):
    """ Export the report to the evaluation folder.

    The name of the report is versioned using today's date
    """
    today = pd.Timestamp.today().strftime('%Y%m%d')
    # Export the report to the evaluation folder write the index as a column named model
    df = df.reset_index()
    df.to_csv(C.EVALUATION_PATH / f'evaluation_{today}.csv', index=False)
def store_recommendations(df, algo_name):
    """ store the recommendation in the rec folder.
    the format should be the following:
    [algo_name] [user_id] [item_id]
    """
    df.to_csv(C.REC_PATH / f'{algo_name}.csv', index=False, header=False)


def load_genres():
    df_genres = pd.read_csv(C.CONTENT_PATH / C.GENRE_FILENAME)
    df_genres = df_genres.set_index(C.ITEM_ID_COL)
    # Split genre strings into list of genres
    df_genres['genres'] = df_genres['genres'].str.split('|')

    # Apply one-hot encoding to the genres
    df_genres = df_genres.genres.str.join('|').str.get_dummies()
    return df_genres




def load_years():
    df_years = pd.read_csv(C.CONTENT_PATH/C.ITEMS_FILENAME)
    # get the year from the title column it's the last 6 character
    df_years['year'] = df_years['title'].str[-6:]
    # get rid of the parenthesis in the year column
    df_years['year'] = df_years['year'].str.replace('(','')
    df_years['year'] = df_years['year'].str.replace(')', '')
    # set movie id as index
    df_years = df_years.set_index(C.ITEM_ID_COL)
    df_years = df_years.drop(["title","genres"],axis=1)
    # get every decade from the year column
    df_years['year'] = df_years['year'].str[2] + "0"
    # make 20 as 0.1 30 as 0.2 and so on
    df_years['year'] = df_years['year'].astype(int)
    df_years['year'] = df_years['year'].replace(00,0.9)
    df_years['year'] = df_years['year'].replace(10,1)
    df_years['year'] = df_years['year'].replace(20,0.1)
    df_years['year'] = df_years['year'].replace(30,0.2)
    df_years['year'] = df_years['year'].replace(40,0.3)
    df_years['year'] = df_years['year'].replace(50,0.4)
    df_years['year'] = df_years['year'].replace(60,0.5)
    df_years['year'] = df_years['year'].replace(70,0.6)
    df_years['year'] = df_years['year'].replace(80,0.7)
    df_years['year'] = df_years['year'].replace(90,0.8)

    print(df_years.columns)
    return df_years

def load_actors():
    df_actors = pd.read_csv("docs/actor_in_common.csv")
    return df_actors

def load_directors():
    df_directors = pd.read_csv("docs/director.csv")
    df_directors.drop(["empty"],axis=1,inplace=True)
    return df_directors
def load_tmdbr():
    df_tmdbr = pd.read_csv("docs/score.csv")
    df_tmdbr = df_tmdbr.set_index(C.ITEM_ID_COL)
    # normalize score between 0 and 1
    df_tmdbr['score'] = df_tmdbr['score']/100

    return df_tmdbr

def load_budget():
    df_budget = pd.read_csv("docs/budget.csv")
    df_budget = df_budget.set_index(C.ITEM_ID_COL)
    # normalize budget between 0 and 1
    max_budget = df_budget['budget'].max()
    min_budget = df_budget['budget'].min()
    df_budget['budget'] = (df_budget['budget'] - min_budget) / (max_budget - min_budget)
    return df_budget







def load_all(): # load all the content data and merge them in a single dataframe
    df_items = load_items()
    df_genres = load_genres()
    df_years = load_years()
    df_score = load_tmdbr()
    df_budget = load_budget()

    df_items = df_items.join(df_genres, how='left')
    df_items.drop(["genres"],axis=1,inplace=True)
    df_items = df_items.join(df_score, how='left')
    df_items = df_items.drop(["title"],axis=1)
    df_items = df_items.join(df_years, how='left')
    df_items = df_items.join(df_budget, how='left')
    df_items.drop(["genres"],axis=1,inplace=True)
    print(df_items.info(),df_score.info(),df_budget.info())


    return df_items

