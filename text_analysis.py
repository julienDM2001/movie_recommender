import math
from typing import Dict
from typing import List
from typing import Optional
from os import sep
from os import walk
import os
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.test.utils import get_tmpfile
import numpy
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
nltk.download("punkt")
nltk.download("stopwords")
from math import sqrt
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud
import matplotlib.pyplot as plt

class Token:
    """
    Class representing a given token. It stores the string representing the token, its identifier and the number of
    documents

    |

    The instance attributes are:

    token_id:
        Identifier of the token.
    token:
        String representing the token.
    docs:
        Identifiers of documents containing the token.
    """

    # -------------------------------------------------------------------------
    token_id: int
    token: str
    docs: List[int]

    # -------------------------------------------------------------------------
    def __init__(self, token_id: int, token: str):
        """
        Constructor.

        :param token_id: Identifier of the token.
        :param token: String representing the token.
        """
        self.token_id = token_id
        self.token = token
        self.docs = []

    def get_idf(self, nb_docs: int) -> float:
        """
        Compute the IDF factor of a token.

        :param nb_docs: Total number of documents in the corpus.
        :return: IDF factor.
        """

        if len(self.docs) == 0:
            return 0.0
        return math.log(float(nb_docs) / float(len(self.docs)))

    # -------------------------------------------------------------------------


class Doc:
    """
    This class represents an instance of a document.

    |

    The instance attributes are:

    url:
        URL of the document (if defined).
    doc_id:
        Identifier of the document.
    text:
        Text of the document to analyse.
    vector:
        Vector representing the document.
    tokens:
        List of tokens i order of appearances. A same token may appear several times.
    """

    # -------------------------------------------------------------------------
    url: Optional[str]
    doc_id: int
    text: str
    vector: numpy.ndarray
    tokens: List[Token]

    # -------------------------------------------------------------------------
    def __init__(self, doc_id: int, text: str, url: Optional[str] = None):
        """
        Constructor.

        :param doc_id:
        :param text: Text of the document (brut).
        :param url: URL of the document (if any).
        """
        self.url = url
        self.doc_id = doc_id
        self.text = text
        self.vector = None
        self.tokens = None


class DocCorpus:
    """
    This class represents a corpus of documents and the corresponding dictionary of tokens contained.

    |

    The instance attributes are:

    docs:
        List of documents.
    tokens:
        Dictionary of tokens (strings are the key).
    ids:
        Dictionary of tokens (identifiers are the key).
    method:
        String representing the method used for analysing ("TF-IDF" or "Doc2Vec").
    nb_dims:
        Number of dimensions of the semantic space.
    stopwords:
        List of stopwords to eliminate from the analysis. By default, it's the classic English list.
    """

    # -------------------------------------------------------------------------
    docs = List[Doc]
    tokens: Dict[str, Token]
    ids: Dict[int, Token]
    method: str
    nb_dims: int
    stopwords: List[str]

    # -------------------------------------------------------------------------
    def __init__(self):
        """
        Constructor.
        """
        self.docs = []
        self.tokens = dict()
        self.ids = dict()
        self.method = "Doc2Vec"
        self.nb_dims = 0
        self.n_tokens = 0
        self.stopwords = stopwords.words('english')

    # -------------------------------------------------------------------------
    def set_method(self, name) -> None:
        """
        Change the parameter.

        :param name: Name of the method.
        """
        self.method = name

    # -------------------------------------------------------------------------
    def add_doc(self, new_doc: str, url: Optional[str] = None) -> None:
        """
        Add a string representing a document to the corpus and provides an
        identifier to the document.

        :param new_doc: New document.
        :param url: URL of the document (if any)
        """
        new_id = len(self.docs)
        self.docs.append(Doc(new_id, new_doc, url))

    # -------------------------------------------------------------------------
    def add_docs(self, docs: List[str]) -> None:
        """
        Add a list of strings representing documents to the corpus. Each document receives an
        identifier.

        :param docs: List of documents.
        """
        for cur_doc in docs:
            self.add_doc(cur_doc)

    # -------------------------------------------------------------------------
    def build_vectors(self) -> None:
        """
        Build the vectors for the documents of the corpus based on the current method.
        """

        if self.method == "Doc2Vec":
            self.build_doc2vec()
        elif self.method == "TF-IDF":
            self.build_tf_idf()
        else:
            raise ValueError("'" + self.method + "': Invalid building method")

    # -------------------------------------------------------------------------


    # -------------------------------------------------------------------------
    def add_token(self, cur_doc: Doc, token_str: str) -> None:
        """Add a token in string format to the Doc Corpus

        Find the identifier of the current token in the dictionary.
        If not present, create a new Token instance

        Attach the token to the current document

        Finally, link the document to the Token object

        :param cur_doc : the current document from which the token is extracted
        :token_str : the token after cleaning steps (stopwords, stemming, ...)

        """
        # Find the identifier of the current token in the dictionary
        if token_str not in self.tokens.keys():
            token_id = len(self.tokens)
            token = Token(token_id, token_str)
            self.tokens[token_str] = token
            self.ids[token_id] = token
            self.n_tokens = len(self.tokens)
        else:
            token = self.tokens[token_str]

        # Add the token
        cur_doc.tokens.append(token)

        # Add a reference count if necessary
        if cur_doc.doc_id not in token.docs:
            token.docs.append(cur_doc.doc_id)

    # -------------------------------------------------------------------------
    def extract_tokens(self) -> None:
        """
        Extract the tokens from the text of the documents. In practice, for each document, the methods
        do the following steps:

        1. The text is transform in lowercase.

        2. The text is tokenised.

        3. Stopwords are removed.

        The method words incrementally. Once a document is treated, it will not be re-treated in successive
        calls.
        """

        # @COMPLETE : create a stemmer
        for cur_doc in self.docs:
            stemmer = SnowballStemmer("english", ignore_stopwords=True)

            if cur_doc.tokens is not None:
                continue
            cur_doc.tokens = []
            text = cur_doc.text

            for extracted_token in nltk.word_tokenize(text):
                token_str_lower = extracted_token.lower()
                token_str = stemmer.stem(token_str_lower)

                #  @COMPLETE : Retains only the stem of non stopwords and punctuation
                if token_str not in stopwords.words("english") and token_str not in [".", ",", "\"", "?", "\'s", "!",";","-","","``","''"]:
                    self.add_token(cur_doc, token_str)



    # -------------------------------------------------------------------------
    def build_tf_idf(self) -> None:
        """
        Build the vectors of the corpus using the TF-IDF approach.
        """


        self.extract_tokens()
        zero_matrix = numpy.zeros((len(self.docs), len(self.tokens)))

        # Step 1: For each document, compute the relative frequencies of each token (TF).
        for cur_doc in range(len(self.docs)):
            vector = dict()  # Dictionary representing a vector of pairs (token_id,nb_occurrences)
            nb_occurrences = 0
            stemmer = SnowballStemmer("english", ignore_stopwords=True)
            text = self.docs[cur_doc].text
            list_word_doc = []

            for extracted_token in nltk.word_tokenize(text):
                token_str_lower = extracted_token.lower()
                token_str = stemmer.stem(token_str_lower)
                list_word_doc.append(token_str)

            count = 0
            for word in self.tokens:
                if word not in list_word_doc:
                    count += 1
                    continue
                a = list_word_doc.count(word)
                vector[word] = a
                zero_matrix[cur_doc][count] = a
                count += 1
            maxi = max(zero_matrix[cur_doc])
            if maxi == 0:
                continue
            for i in range(len(zero_matrix[cur_doc])):
                zero_matrix[cur_doc][i] = zero_matrix[cur_doc][i] / maxi


        dif_vector = dict()
        # Step 2: Build the TF-IDF vectors by multiplying the relative frequencies by the IDF factor.
        self.nb_dims = self.n_tokens
        for cur_doc in self.docs:
            text = cur_doc.text
            which_doc = []
            for extracted_token in nltk.word_tokenize(text):
                token_str_lower = extracted_token.lower()
                token_str = stemmer.stem(token_str_lower)
                if token_str not in dif_vector and token_str not in which_doc:
                    dif_vector[token_str] = 1
                    which_doc.append(token_str)
                if token_str in dif_vector and token_str not in which_doc:
                    dif_vector[token_str] += 1
                    which_doc.append(token_str)
        for word in dif_vector:
            dif_vector[word] = math.log(len(self.docs) / dif_vector[word])
        for i in range(len(zero_matrix)):
            for y in range(len(zero_matrix[i])):
                if zero_matrix[i][y] == 0 or list(dif_vector.values())[y] == 0:
                    continue
                zero_matrix[i][y] = zero_matrix[i][y] / list(dif_vector.values())[y]
        for i in range(len(self.docs)):
            self.docs[i].vector = zero_matrix[i]
        return zero_matrix



    # -------------------------------------------------------------------------
    def build_doc2vec(self) -> None:
        """
        Build the vectors using the doc2vec approach.
        """

        self.extract_tokens()
        corpus = []
        for doc in self.docs:
            tokens = []
            for token in doc.tokens:
                tokens.append(token.token)
            corpus.append(tokens)

        corpus = [
            TaggedDocument(words, ['d{}'.format(idx)])
            for idx, words in enumerate(corpus)
        ]
        model = Doc2Vec(corpus, vector_size=round(sqrt(len(corpus))), window=2, min_count=1, workers=4)
        fname = get_tmpfile("my_doc2vec_model")
        model.save(fname)
        model = Doc2Vec.load(fname)
        for i in range(len(corpus)):
            for y in corpus[i]:
                print(y)
                vector = model.infer_vector(y)
                self.docs[i].vector = vector
        return [self.docs[i].vector for i in range(len(corpus))]

class TokenSorter:
    """
    Class to sort a list of tokens by a certain value.
    |

    The instance attributes are:

    tokens:
        List of tokens to sort.
    reverse:
        Must the token be ranked descending (False) or ascending (True)
    """

    # -------------------------------------------------------------------------
    class TokenRef:
        """
        Class to represent a reference to a token.
        """

        # ---------------------------------------------------------------------
        token: Token
        value: float

        # ---------------------------------------------------------------------
        def __init__(self, token: Token, value: float):
            self.token = token
            self.value = value

    # -------------------------------------------------------------------------
    tokens: List[TokenRef]
    reverse: bool

    # -------------------------------------------------------------------------
    def __init__(self):
        """
        Constructor.
        """

        self.tokens = []
        self.reverse = False

    # -------------------------------------------------------------------------
    def build(self, tokens, value, reverse: bool) -> None:
        """
        Build the list of token to sort.

        :param tokens: Tokens to sort.
        :param value: Lambda function that will be used to build the value associated to each token to sort.
        :param reverse: Should the token be sorted in descending (True) of ascending (False) order.
        """

        for token in tokens.values():
            self.add_token(token, value(token))
        self.reverse = reverse
        self.sort()

    # -------------------------------------------------------------------------
    def add_token(self, token: Token, value: float) -> None:
        """
        Add a token to the list.

        :param token: Token to add.
        :param value: Value that will be used to sort the tokens.
        """

        self.tokens.append(TokenSorter.TokenRef(token=token, value=float(value)))

    # -------------------------------------------------------------------------
    def sort(self) -> None:
        """
        Sort the tokens.
        """

        self.tokens.sort(reverse=self.reverse, key=lambda token: token.value)

    # -------------------------------------------------------------------------
    def get_token(self, pos: int) -> str:
        """
        Get a given token of the list.

        :param pos: Position of the token in the list.
        :return: String representing the token.
        """

        return self.tokens[pos].token.token

    # -------------------------------------------------------------------------
    def get_value(self, pos: int) -> str:
        """
        Get a value of a given token in the list.

        :param pos: Position of the token in the list.
        :return: Value of the token used for the sorting.
        """

        return self.tokens[pos].value

    # -------------------------------------------------------------------------
    def print(self, title: str, nb : int) -> None:
        """
        Print a given number of top ranked tokens with a title and their values.

        :param title: Title to print.
        :param nb:  Number of tokens to print.
        """
        print(title)
        if nb > len(self.tokens):
            nb = len(self.tokens)
        for i in range(0,nb):
            print(self.get_token(i)) #self.get_value(i))


def print_matrix(name: str, matrix: numpy.ndarray) -> None:
    """
    Simple method to print a little matrix nicely.

    :param name:  Name of the matrix.
    :param matrix:  Matrix to print.
    """
    nb_lines = matrix.shape[0]
    nb_cols = matrix.shape[1]
    spaces = " " * (len(name) + 1)
    title_line = nb_lines % 2
    for i in range(0, nb_lines):
        if i == title_line:
            print(name + "=", end="")
        else:
            print(spaces, end="")
        print("( ", end="")
        for j in range(0, nb_cols):
            print("{:.3f}".format(matrix[i, j]), end=" ")
        print(")", )


def create_corpus(path: str) -> DocCorpus:
    """
    From a list of docs located at path, create a corpus

    A DocCorpus document is build and populated with all the "doc" documents
    located at the path

    :param path : string description of the path

    :return : DocCorpus representing the corpus of all the documents
    """
    # Instantiate a DocCorpus object
    the_corpus = DocCorpus()

    # Look for all the files in a directory
    files = []
    dir_to_analyse = path
    for (_, _, file_names) in walk(dir_to_analyse):

        files.extend(file_names)
        break

    # Add the context to the corpus
    for doc_to_analyse in files:
        # Treat only files beginning with "doc"
        if doc_to_analyse == "links.txt" or doc_to_analyse == "actors.csv":
            continue

        filename = dir_to_analyse + sep + doc_to_analyse
        file = open(file=filename, mode="r", encoding="utf-8")
        the_corpus.add_doc(file.read(), filename)

    return the_corpus


# create a corpus for each genre in film_by_genre directory


#create a function that loop throug the film_by_genre directory and print each directory name
all_corpus = []
x = []
for (_, dir_names, _) in walk("./film_by_genre"):

    for i in dir_names:
        if i not in ["Children","(no genres listed)", "IMAX", "Film-Noir","Sci-Fi","Musical","general_overview"]:
            x.append(i)
    break





for i in x:
    all_corpus.append(create_corpus("film_by_genre/"+i))
x.append("top_500")
x.append("top_500_tmdb")
all_corpus.append(create_corpus("top_500"))
all_corpus.append(create_corpus("top_500_tmdb"))
print(all_corpus)

# create a function that loop throug all_corpus and print the most appearing tokens and mos discriment tokens for each genre
def all_analysis():
    for i in range(len(all_corpus)):
        print("Genre: ", x[i])
        all_corpus[i].extract_tokens()
        sort_by_docs = TokenSorter()
        sort_by_docs.build(all_corpus[i].tokens, value=lambda token: len(token.docs), reverse=True)
        sort_by_docs.print("Most appearing tokens", 20)
        print("now the most discriminating tokens \n\n\n")
        sort_by_iDF = TokenSorter()
        sort_by_iDF.build(tokens=all_corpus[i].tokens, value=lambda token: token.get_idf(len(all_corpus[i].docs)), reverse=True)
        sort_by_iDF.print(title="Most discriminant  tokens (idf):",nb=20)


all_analysis()


the_corpus = create_corpus("./docs")# Create a corpus instance
the_corpus.extract_tokens() # Extract the tokens from the corpus


# -------------------------------------------------------------------------------------------------
# Sort the tokens by the number of documents in which they appear

sort_by_docs = TokenSorter()
sort_by_docs.build(tokens=the_corpus.tokens, value=lambda token: len(token.docs), reverse=True)
sort_by_docs.print(title="Most appearing tokens (Nb Documents):",nb=20)
# create a wordcloud with the token
print("now the most discriminating tokens \n\n\n")

# Sort the tokens by their idf factor
sort_by_iDF = TokenSorter()
sort_by_iDF.build(tokens=the_corpus.tokens, value=lambda token: token.get_idf(len(the_corpus.docs)), reverse=True)
sort_by_iDF.print(title="Most discriminant  tokens (idf):",nb=20)

# -------------------------------------------------------------------------------------------------
# create a function that creat an all_overview_genre.txt file that concanate all the overview doc for each genre in film_by_genre
# def create_overview_genre(path: str) -> DocCorpus:
#     """loop throug the film_by_genre directory and add the text of each of the doc in that directory in a new doc called overview_genre.txt"""
#     # Look for all the files in a directory
#     files = []
#     for (_, _, file_names) in walk(path):
#         files.extend(file_names)
#         break
#     # Add the context to the corpus
#     for doc_to_analyse in files:
#         # Treat only files beginning with "doc"
#         if doc_to_analyse == "links.txt" or doc_to_analyse == "actors.csv":
#             continue
#
#         filename = path + sep + doc_to_analyse
#         file = open(file=filename, mode="r", encoding="utf-8")
#         print(filename)
#         overview = open(file="./film_by_genre/general_overview/" + str(i) +" overview", mode="a", encoding="utf-8")
#         overview.write(file.read())
#         overview.close()
#
# for i in x:
#     create_overview_genre("./film_by_genre/"+i)





def corpus_analysis(corpus: DocCorpus, method: str, path) -> None:
    """
    Calculate and display the cosine similarity between every pair of document for a given vectorization method

    :param corpus: Corpus to analyse.
    :param method: Method to use for the analysis.
    """

    print("\n---- " + method + " ----")
    corpus.set_method(method)
    if method == "TF-IDF":
        matrix = corpus.build_tf_idf()
    else:
        matrix = corpus.build_doc2vec()

    # @COMPLETE : compute cosine similarity between every vector of the matrix


    if method == "TF-IDF":
        with open(path, "w") as f:
            for i in range(0, len(corpus.docs) - 1):
                print(i)
                    # Take a vector and build a two dimension matrix needed by cosine_similarity
                vec1 = matrix[i].reshape(1, -1)

                for j in range(i + 1, len(corpus.docs)):
                    # Take a vector and build a two dimension matrix needed by cosine_similarity
                    vec2 = matrix[j].reshape(1, -1)

                        # Retrieve name of the docs
                    url_i = corpus.docs[i].doc_id
                    url_j = corpus.docs[j].doc_id

                        # Compute and display the similarity
                    if round(cosine_similarity(vec1, vec2)[0][0], 3) > 0.001:
                        f.write(str(url_i))
                        f.write(",")
                        f.write(str(url_j))
                        f.write(",")
                        f.write(str(round(cosine_similarity(vec1,vec2)[0,0],3)))
                        f.write("\n")
    if method == "Doc2Vec":
        with open(path, "w") as f:
            for i in range(0, len(corpus.docs) - 1):
                print(i)
                    # Take a vector and build a two dimension matrix needed by cosine_similarity
                vec1 = matrix[i].reshape(1, -1)

                for j in range(i + 1, len(corpus.docs)):
                        # Take a vector and build a two dimension matrix needed by cosine_similarity
                    vec2 = matrix[j].reshape(1, -1)

                        # Retrieve name of the docs
                    url_i = corpus.docs[i].doc_id
                    url_j = corpus.docs[j].doc_id
                    if round(cosine_similarity(vec1, vec2)[0][0], 3) > 0.001:
                        f.write(str(url_i))
                        f.write(",")
                        f.write(str(url_j))
                        f.write(",")
                        f.write(str(round(cosine_similarity(vec1,vec2)[0,0],3)))
                        f.write("\n")








# -----------------------------------------------------------------------------------------------------------
#corpus_analysis(corpus=the_corpus, method="TF-IDF")
#corpus_analysis(corpus=the_corpus, method="Doc2Vec")
#store the cosine similarity between each pair of doc in a csv file


#new_corpus = create_corpus("film_by_genre/general_overview")

#corpus_analysis(corpus=new_corpus, method="TF-IDF", genre=True)
#corpus_analysis(corpus=new_corpus, method="Doc2Vec", genre=True)
