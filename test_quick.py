import pandas as pd
from constants import Constant as C
pd.set_option('display.max_columns', None)

user_id = 1
df = pd.read_csv(f"{C.REC_PATH}/top_n_recs.csv")
df_top10 = pd.read_csv("docs/score.csv")
genre = "Children"
n = 10












def show_movies():
    # filter the dataframe by the user id
    df_user = df[df["user_id"] == user_id]
    # filter the dataframe by the model selected
    df_model = df_user[df_user["algo"] == "ContentBased"]
    # filter the dataframe by the number of movies selected
    df_n = df_model.head(n)
    # filter the dataframe by the genre selected

    # merge the dataframe with the movies.csv file to get the movie title
    df_movie = pd.read_csv(f"{C.CONTENT_PATH}/{C.ITEMS_FILENAME}")
    df_merge = pd.merge(df_n, df_movie, on="movieId")
    # filter the dataframe by the genre selected if all is selected, show all the movies
    if genre != "All":
        if len(df_merge[df_merge["genres"].str.contains(genre)]) > n :
            df_merge = df_merge[df_merge["genres"].str.contains(genre)]
        else: # get the film of the same genre first and fill the rest  with the highest score
            df_merge = df_merge[df_merge["genres"].str.contains(genre)]
            df_merge = df_merge.append(df_top10[df_top10["genres"].str.contains(genre)],ignore_index=True)
    # create a number of column equal to the number of movies selected

    return df_merge

print(show_movies().head())
