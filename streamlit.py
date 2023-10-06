import streamlit as st
import pandas as pd
from constants import Constant as C
import random as rd
from PIL import Image
# create a web app with streamlit
df = pd.read_csv(f"{C.REC_PATH}/top_n_recs.csv")
df_top10 = pd.read_csv("docs/score.csv")
df_top10 = df_top10.sort_values(by="score", ascending=False)

# create a dataframe with the top 10 movies of the last 10 years
df_top10_last10 = pd.read_csv("docs/score2.csv")
df_top10_last10 = df_top10_last10.sort_values(by="score", ascending=False)
# take only the movies of the most recent years
df_top10_last10 = df_top10_last10[df_top10_last10["year"] >= 2016]



st.set_page_config(layout="wide")
st.title("Welcome to the Movie Recommender System")
st.subheader("This is a simple movie recommender system that uses the MovieLens dataset")

# create an Image object to show the logo
st.sidebar.title("Movie Recommender System")
# on the side bar, create a select box for the user to select the model
dicm = {"Based on what you like": "ContentBased", "Based on similar user": "UserBased", "Based on magical math": "SVD"}
model = st.sidebar.selectbox("Select Model", {"Based on what you like": "ContentBased", "Based on similar user": "UserBased", "Based on magical math": "SVD"})
# store the model selected in a variable
# show the movie accorded to the model selected
st.sidebar.subheader("Show Movie")
# create a slider to select the number of movies to show
n = st.sidebar.slider("Number of movies", 1, 20, 5)
# create a button to show the movies
show = st.sidebar.button("Show Movies")
# create a sidebar to choose the user id
st.sidebar.subheader("Choose User ID")

# link the choose user id sidebar to the user id variable in the top_n_recs.csv
user_id = st.sidebar.selectbox("User ID", df["user_id"].unique())
genre = st.sidebar.selectbox("Select Genre", ("All", "Action", "Adventure", "Animation", "Children", "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"))




df_movie = pd.read_csv(f"{C.CONTENT_PATH}/{C.ITEMS_FILENAME}")

# create a function to show the movies store in the C.REC_PATH/top_n_recs.csv file
def show_movies():
    # filter the dataframe by the user id
    df_user = df[df["user_id"] == user_id]
    # filter the dataframe by the model selected
    df_model = df_user[df_user["algo"] == dicm[model]]
    # merge the dataframe with the movies.csv file to get the movie title
    df_movie = pd.read_csv(f"{C.CONTENT_PATH}/{C.ITEMS_FILENAME2}")
    df_merge = pd.merge(df_model, df_movie, on="movieId")
    # filter the dataframe by the genre selected if all is selected, show all the movies
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


    # create a number of column equal to the number of movies selected
    st.subheader(f"Top {n} movies recommended for user {user_id}  {model}")
    columns = st.columns(n)
    for i in range(n):
        with columns[i]:
            movieId = int(df_merge.iloc[i]["movieId"])
            image = Image.open(f"{C.POSTERS_PATH}/{movieId}.jpg")
            resized_image = image.resize((200, 300))
            columns[i] =st.image(resized_image, width=200)


# create a select genre to filter the movies must include all the genre store in the movie.csv file
# create a sidebar





# create a subheader to show the top 8 movies recommended by the model choosen
st.subheader(f"based on what you watched, we recommend you these movies")
first_column = st.columns([1,1,1,1,1,1,1,1])
for i in range(8):
    with first_column[i]:
        # find the first i movie reccommended by the content based model for the user
        model1 = "ContentBased"
        movie = int(df[df["user_id"]==user_id][df["algo"]==model1].iloc[i]["movieId"])
        image = Image.open("posters/" + str(movie) + ".jpg")
        resized_image = image.resize((250, 400))
        st.image(resized_image, width=250,caption=df_movie[df_movie["movieId"]==movie]["title"].values[0])









st.subheader("Top 10 Movies on TMDB",anchor=False)
columns = st.columns([1 for i in range(8)])
for i in range(8):
    with columns[i]:
        # find tthe first movie in df_top10
        movie = int(df_top10.iloc[i]["movieId"])
        image = Image.open("posters/" + str(movie) + ".jpg")
        resized_image = image.resize((250, 400))
        st.image(resized_image, width=250,caption=df_movie[df_movie["movieId"]==movie]["title"].values[0])


# create a column to show the top 10 movie on tmdb of the last 10 years
st.subheader("Top 10 Movies on TMDB of the moment",anchor=False)
columns1 = st.columns([1 for i in range(8)])
for i in range(8):
    with columns1[i]:
        # find the first movie in df_top10
        movie = int(df_top10_last10.iloc[i]["movieId"])
        image = Image.open("posters/" + str(movie) + ".jpg")
        resized_image = image.resize((250, 400))
        st.image(resized_image, width=250,caption=df_movie[df_movie["movieId"]==movie]["title"].values[0])




# create a subheader to show the original movie

st.subheader("original movie",anchor=False)
columns2 = st.columns([1 for i in range(8)])
for i in range(8):
    list_of_rand = []
    with columns2[i]:
        # select a random movie ammong the movieId in df_movie
        rand = rd.choice(df_movie["movieId"])
        while rand in list_of_rand:
            rand = rd.choice(df_movie["movieId"])
        list_of_rand.append(rand)
        image = Image.open(f"{C.POSTERS_PATH}/{rand}.jpg")
        resized_image = image.resize((250, 400))
        st.image(resized_image, width=250,caption=df_movie[df_movie["movieId"] == rand]["title"].values[0])

if show:
    show_movies()