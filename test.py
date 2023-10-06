import pandas as pd
from constants import Constant as C
pd.set_option('display.max_columns', None)

df = pd.read_csv(f"{C.REC_PATH}/top_n_recs.csv")
genre = "All"
df_user = df[df["user_id"] == 1]
# filter the dataframe by the model selected
df_model = df_user[df_user["algo"] == "ContentBased"]
# filter the dataframe by the number of movies selected
df_top10 = pd.read_csv("docs/score.csv")
df_top10 = df_top10.sort_values(by="score", ascending=False)
n = 10
# filter the dataframe by the genre selected

# merge the dataframe with the movies.csv file to get the movie title
df_movie = pd.read_csv(f"{C.CONTENT_PATH}/{C.ITEMS_FILENAME2}")
df_merge = pd.merge(df_model, df_movie, on="movieId")

if genre != "All":
    # check if the binary column of the genre selected has more than n movies
    if len(df_merge[df_merge[f"genres_{genre}"] == 1]) >= n:
        df_merge = df_merge[df_merge[f"genres_{genre}"] == 1]
    else:
        # get the all the movies of the genre selected and fill the rest with df_top10 movies
        df_genre = df_merge[df_merge[f"genres_{genre}"] == 1]
        df_top = pd.read_csv("docs/score.csv")
        df_top = df_top.sort_values(by="score", ascending=False)
        # filter the df_top by genre note the genre column as the fromat genre1|genre2|genre3...
        df_top["genres"] = df_top["genres"].str.split("|")
        df_top = df_top.explode("genres")
        df_top = df_top[df_top["genres"] == genre]
        df_top = df_top.head(n - len(df_genre))
        df_merge = pd.concat([df_genre, df_top])
for i in range(n):
    print(int(df_merge.iloc[i]["movieId"]))