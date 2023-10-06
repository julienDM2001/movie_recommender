import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import bs4
import requests
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import seaborn as sns
from typing import List, Dict
import codecs
from constants import Constant as C
import re
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept-Language': 'en-US,en;q=0.9',
    'Authorization': 'Bearer YourAccessToken',
    # Add any other headers you need
}


#DORYAN
link = pd.read_csv(f"{C.CONTENT_PATH}/links.csv")
movie = pd.read_csv(f"{C.CONTENT_PATH}/{C.ITEMS_FILENAME}")

class Actor:
    """
    This class represents an actor.

    |

    The instance attributes are:

    actor_id:
        Identifier of the actor.

    name:
        Name of the actor.

    movies:
        List of movies in which the actor has played.
    """

    # -------------------------------------------------------------------------
    actor_id: int
    name: str
    movies: List["Movie"]

    # -------------------------------------------------------------------------
    def __init__(self, actor_id: int, name: str):
        """
        Constructor.

        :param actor_id: Identifier of the actor.
        :param name: Name of the actor.
        """

        self.actor_id = actor_id
        self.name = name
        self.movies = []

class Movie:
    """
    This class represents a movie_to_analyse.

    |

    The instance attributes are:

    movie_id:
        Identifier of the movie_to_analyse.

    name:
        Name of the movie_to_analyse in the IMDb database.

    actors:
        List of actors who have played in the movie_to_analyse.

    summary:
        Summary of the movie_to_analyse.
    """

    # -------------------------------------------------------------------------
    movie_id: int
    name: str
    actors: List[Actor]
    summary: str

    # -------------------------------------------------------------------------
    def __init__(self, movie_id: int, name: str):
        """
        Constructor.

        :param movie_id: Identifier of the movie_to_analyse.
        :param name: Name fo the movie_to_analyse.
        """

        self.movie_id = movie_id
        self.name = name
        self.actors = []
        self.summary = ""
        
    def __repr__(self):
        return "For info movie class is made up of {} movie_id, {} name, {} actors, {} summary \n".format(self.movie_id,self.name,self.actors,self.summary)

class Parser:
    genre_list = dict()
    main_genre = dict()
    score_film = dict()
    budget = dict()
    director = dict()
    """

    |

    The instance attributes are:

    output:
        Directory where to store the resulting data.

    basic_url:
        Begin of the URL used to retrieve the HTML page of a movie_to_analyse.

    actors:
        Dictionary of actors (the identifiers are the key).

    actors:
        Dictionary of actors (the names are the key).

    movies:
        Dictionary of movies (the identifiers are the key).
    """

    # -------------------------------------------------------------------------
    output: str
    basic_url: str
    actors: Dict[int, Actor]
    actors_by_name: Dict[str, Actor]
    movies: Dict[int, Movie]

    # -------------------------------------------------------------------------
    def __init__(self, output: str, basic_url: str) -> None:
        """
        Initialize the parser.

        :param output: Directory where to store the results.
        :param basic_url: Beginning part of the URL of a movie_to_analyse page.
        """

        self.output = output + os.sep
        if not os.path.isdir(self.output):
            os.makedirs(self.output)
        self.basic_url = basic_url
        self.actors = dict()
        self.actors_by_name = dict()
        self.movies = dict()
        self.genre = list()

    # -------------------------------------------------------------------------
    def extract_data(self, movie: str) -> None:
        """
        Extract the "useful" data from the page. In practice, the following steps are executed:

        1. Build the URL of the movie_to_analyse page.

        2. Create a new Movie instance and add it to the list.

        3. Download the HTML page and use an instance of BeautifulSoup to parse.

        4. Extract all "div" tags and analyze those of the class "summary_text" (summary of the movie_to_analyse) and
        "credit_summary_item" (directors, producers, actors, etc.).

        :param movie: Analyzed movie_to_analyse.
        """
        print(movie)
        url = self.basic_url + movie
        movie1 = movie


        doc_id = len(self.movies) + 1  # First actor_id = 1
        movie = Movie(doc_id, movie)
        self.movies[doc_id] = movie
 
        
        # Download the HTML using the requests library, check the status-code and extract the text
        ## @COMPLETE : use the requests library here, get the response and extract the content

        response = requests.get(url,headers=headers)
        content = response.content

        # Download the HTML and parse it through Beautifulsoup
        soup = bs4.BeautifulSoup(content, "html.parser")
        
        # Extract infos
        self.extract_summary(movie, soup)
        self.extract_actors(movie, soup)
        self.extract_genre(movie, soup)
        self.extract_score(movie1, soup)
        self.extract_budget(movie1, soup)
        self.extract_director(movie1, soup)

    
    # -------------------------------------------------------------------------
    def extract_summary(self, movie, soup) -> None:
        """
        This function extract the summary from a movie/tv-show
        It use the find_all method of BeautifulSoup to find the "overview" class
        """
        divs = soup.find_all("div")
        for div in divs:
            div_class = div.get("class")
            if div_class is not None:
                if 'overview' in div_class:
                    movie.summary = div.text

    def extract_score(self, movie, soup) -> None:
        """
        This function extract the score from a movie/tv-show
        It use the find_all method of BeautifulSoup to find the "ratingValue" class
        """
        divs = soup.find_all("div")
        for div in divs:
            div_class = div.get("class")
            if div_class is not None:
                if 'user_score_chart' in div_class:
                    # get the calue store in data-percent in the div
                    score = div.get("data-percent")


                    Parser.score_film[movie] = score

        
        
    # -------------------------------------------------------------------------
    def extract_actors(self, movie, soup) -> None:
        """
        This function extract the list of actors displayed for a specific movie/tv-show
        It use the select method of BeautifulSoup to extract actors displayed on the page.
        Actor are defined in people scroller cards
        """

        soup_results = soup.select("ol[class='people scroller'] li[class='card'] p a")
        actors = [soup_result.text for soup_result in soup_results]

        # Store actors in class dictionaries
        for actor in actors:
            if actor not in self.actors_by_name.keys():
                actor_id = len(self.actors) + 1  # First actor_id = 1
                new_actor = Actor(actor_id, actor)
                self.actors[actor] = new_actor
                self.actors_by_name[actor] = new_actor
            self.actors_by_name[actor].movies.append(movie)
            movie.actors.append(self.actors_by_name[actor])





    def extract_genre(self, movie, soup) -> None:
        """this function extract the genre of a movie/tv-show and store it in a dictionary with the movie_id as key"""
        divs = soup.find_all("div")
        for div in divs:
            div_class = div.get("class")


            if div_class is not None:
                if 'facts' in div_class:
                    for span in div.find_all('span'):
                        if "genres" in span.attrs['class']:
                            split_text = span.get_text(strip=True)
                            Parser.genre_list[movie.movie_id] = split_text + "\n"
                            new_split = split_text.split(",")

                            Parser.main_genre[movie.movie_id] = new_split[0] + "\n"


    def extract_budget(self, movie, soup) -> None:
        """this function extract the budget of a movie/tv-show and store it in a dictionary with the movie_id as key"""
        divs = soup.find_all("div")
        for div in divs:
            div_class = div.get("class")
            if div_class is not None:

                if "no_bottom_pad" in div_class:
                    for tag in div.find_all('p'): # find the budget in p tag
                        # find the first element that start with $ and store it in budget keep all the zeros after the ,
                        budget = re.findall(r'\$\d{1,3}(?:,\d{3})*(?:\.\d+)?(?!\d)', tag.text)

                        if budget: # store the budget witout the dollar sign
                            budget = budget[0][1:].replace(",", "")
                            Parser.budget[movie] = budget
                            break


    def extract_director(self, movie, soup) -> None:
        """this function extract the director of a movie/tv-show and store it in a dictionary with the movie_id as key"""
        divs = soup.find_all("div")
        for div in divs:
            div_class = div.get("class")
            if div_class is not None:
                if 'header_info' in div_class:
                    for i in div.find_all('li'):
                        if "Director" in i.text:
                            director = i.text.replace("Director: ", "")
                            director = director.replace("\n", ",")
                            director = director.strip()
                            # erase the text after the word director
                            director = director.split("Director,")[0]
                            Parser.director[movie] = director
                            break















    # -------------------------------------------------------------------------
    def write_files(self) -> None:
        """
        Write all the file. Three thinks are done:

        1. For each document, create a file (doc*.txt) that contains the summary and the name of
        the actors.

        2. Create a CSV file "actors.csv" with all the actors and their identifiers.

        3. Build a matrix actors/actors which elements represent the number of times 
        two actors are playing in the same
        movie_to_analyse.

        4. Create a CSV file "links.txt" that contains all the pairs of actors having played together.
        """

        # Write the clean text
        for movie in self.movies.values():
            if len(movie.actors) <2 :
                continue
            # Create a file for each movie with it's summary in the movie_sum directory create a file if it doesn't exist
            with open("movie_sum" + os.sep + "doc" + str(movie.movie_id) + ".txt", "w", encoding="utf-8") as file:
                file.write(movie.summary + "\n")



        # Write the list of actors")
        actors_file = codecs.open(self.output + "actors.csv", 'w', "utf-8")
        for actor in self.actors.values():
            actors_file.write(str(actor.actor_id) + ',"' + actor.name + '"\n')

        # Build the matrix actors/actors
        matrix = np.zeros(shape=(len(self.actors), len(self.actors)))
        for movie in self.movies.values():
            for i in range(0, len(movie.actors) - 1):
                for j in range(i + 1, len(movie.actors)):
                    # ! Matrix begins with 0, actors with 1
                    matrix[movie.actors[i].actor_id - 1, movie.actors[j].actor_id - 1] += 1
                    matrix[movie.actors[j].actor_id - 1, movie.actors[i].actor_id - 1] += 1

        # Write only the positive links
        links_file = codecs.open(self.output + "links.txt", 'w', "utf-8")
        for i in range(0, len(self.actors) - 1):
            for j in range(i + 1, len(self.actors)):
                weight = matrix[i, j]
                if weight > 0.0:
                    # ! Matrix begins with 0, actors with 1
                    links_file.write(str(i + 1) + "," + str(j + 1) + "," + str(weight) + "\n")

        # count the number of actor in common between two movies and store it in a matrix
        dico = {}
        for movie in self.movies.values(): # for each movie count the number of actor in common with the other movies
            for movie2 in self.movies.values():
                if movie.movie_id != movie2.movie_id:
                    count = 0
                    for actor in movie.actors:
                        if actor in movie2.actors:
                            count += 1
                    dico[(movie.movie_id, movie2.movie_id)] = count
        # write the dico in a csv file
        with open("docs/actor_in_common.csv", 'w', newline='') as f:
            for key, value in dico.items():
                f.write("%s,%s,%s\n" % (key[0], key[1], value))

        # write the score of each movie
        score_file = codecs.open(self.output + "score.csv", 'w', "utf-8")
        for movie_id in Parser.score_film.keys():
            score_file.write(str(movie_id) + "," + str(Parser.score_film[movie_id]) + "\n")

        # wrtie the budget of each movie
        budget_file = codecs.open(self.output + "budget.csv", 'w', "utf-8")
        for movie_id in Parser.budget.keys():
                budget_file.write(str(movie_id) + "," + str(Parser.budget[movie_id]) + "\n")

        # for each movie write it's actors in a csv file
        with open("docs/actors_by_movie.csv", 'w', newline='') as f:
            for movie in self.movies.values():
                for actor in movie.actors:
                    f.write("%s,%s\n" % (movie.movie_id, actor.actor_id))

        # write the director of each movie
        director_file = codecs.open(self.output + "director.csv", 'w', "utf-8")
        for movie_id in Parser.director.keys():
            director_file.write(str(movie_id)  + str(Parser.director[movie_id]) + "\n")










list_name = []
list_id = []
for i in movie.title:
     list_name.append(i)
for y in link.tmdbId:
     list_id.append(str(y))
list_movies = list(zip(list_name,list_id))

# create an empty file for each movie in the movie_sum directory


# ----------------------------------------------------------------------------------------
# Initialize a list of movies to download
basic_url_to_analyze = 'https://www.themoviedb.org/movie/'
dir_docs = "docs/"


# -----------------------------------------------------------------------------------------

parser = Parser(dir_docs, basic_url_to_analyze)
for movie_label, movie_id in list_movies:
    parser.extract_data(movie_id)
parser.write_files()

# -----------------------------------------------------------------------------------------
# for all the coma except the first one of each line in the budget.csv file replace it by  nothing



