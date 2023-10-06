# standard library imports
from collections import defaultdict
from sklearn.model_selection import train_test_split
# third parties imports
import numpy as np
import random as rd
from surprise import AlgoBase
from surprise import KNNWithMeans
from surprise import SVD
from loaders import *
from surprise.prediction_algorithms import PredictionImpossible
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from constants import Constant as C
import pandas as pd
import heapq
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.metrics import mean_squared_error

def get_top_n(predictions, n):
    """Return the top-N recommendation for each user from a set of predictions.
    Source: inspired by https://github.com/NicolasHug/Surprise/blob/master/examples/top_n_recommendations.py
    and modified by cvandekerckh for random tie breaking

    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.
    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    """

    rd.seed(0)

    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        rd.shuffle(user_ratings)
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]
    return top_n


# First algorithm
class ModelBaseline1(AlgoBase):
    def __init__(self):
        AlgoBase.__init__(self)

    def estimate(self, u, i):
        return 2


# Second algorithm
class ModelBaseline2(AlgoBase):
    def __init__(self):
        AlgoBase.__init__(self)

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)
        rd.seed(0)

    def estimate(self, u, i):
        return rd.uniform(self.trainset.rating_scale[0], self.trainset.rating_scale[1])


# Third algorithm
class ModelBaseline3(AlgoBase):
    def __init__(self):
        AlgoBase.__init__(self)

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)
        self.the_mean = np.mean([r for (_, _, r) in self.trainset.all_ratings()])

        return self

    def estimate(self, u, i):
        return self.the_mean


# Fourth Model
class ModelBaseline4(SVD):
    def __init__(self):
        SVD.__init__(self, n_factors=100,verbose = True)

# fifth model
class ContentBased(AlgoBase):
    def __init__(self, features_method="all", regressor_method="linear_regression"):
        AlgoBase.__init__(self)
        self.regressor_method = regressor_method
        self.content_features = self.create_content_features(features_method)

    def create_content_features(self, features_method):
        """Content Analyzer"""
        df_items = load_items()
        if features_method is None:
            df_features = None
        elif features_method == "title_length":  # a naive method that creates only 1 feature based on title length
            df_features = df_items[C.LABEL_COL].apply(lambda x: len(x)).to_frame('n_character_title')
        elif features_method == "all":
            df_features = load_all()
            print(df_features.info())
        elif features_method == "year":
            df_features = load_years()
        elif features_method == "genre":
            df_features = load_genres()
        elif features_method == "score":
            df_features = load_tmdbr()
        elif features_method == "budget":
            df_features = load_budget()



        else:  # (implement other feature creations here)
            raise NotImplementedError(f'Feature method {features_method} not yet implemented')
        return df_features

    def fit(self, trainset):
        """Profile Learner"""
        AlgoBase.fit(self, trainset)

        # Preallocate user profiles
        self.user_profile = {u: None for u in trainset.all_users()}
        self.user_profile_explain = {u: None for u in trainset.all_users()}

        if self.regressor_method == 'random_score':
            pass

        elif self.regressor_method == 'random_sample':
            for u in self.user_profile:
                self.user_profile[u] = [rating for _, rating in self.trainset.ur[u]]

        elif self.regressor_method == 'linear_regression':
            feature_names = self.content_features.columns.tolist()

            for u in self.user_profile:
                inner_item_ids, user_ratings = zip(*self.trainset.ur[u])
                df_user = pd.DataFrame({"item_id": inner_item_ids, "user_ratings": user_ratings})

                # Map inner_item_id to raw_item_id
                df_user["item_id"] = df_user["item_id"].map(self.trainset.to_raw_iid)

                # Merge user ratings with content features
                df_user = df_user.merge(self.content_features, how='left', left_on='item_id', right_index=True)

                # Prepare X and y for the linear regression model
                X = df_user[feature_names].values
                y = df_user['user_ratings'].values

                # Fit the linear regression model
                linear_model = LinearRegression(fit_intercept=False)
                linear_model.fit(X, y)

                # Assign the linear regressor to each user
                self.user_profile[u] = linear_model
                self.user_profile_explain[u] = linear_model.coef_

            # (implement here the regressor fitting)
        if self.regressor_method == 'random_forest':
            feature_names = self.content_features.columns.tolist()
            for u in self.user_profile:
                inner_item_ids, user_ratings = zip(*self.trainset.ur[u])
                df_user = pd.DataFrame({"item_id": inner_item_ids, "user_ratings": user_ratings})

                # Map inner_item_id to raw_item_id
                df_user["item_id"] = df_user["item_id"].map(self.trainset.to_raw_iid)

                # Merge user ratings with content features
                df_user = df_user.merge(self.content_features, how='left', left_on='item_id', right_index=True)

                # Prepare X and y for the random forest model
                X = df_user[feature_names].values
                y = df_user['user_ratings'].values

                # Fit the random forest model make sure to make it run fast
                random_forest = RandomForestRegressor(n_estimators=5, max_depth=10, n_jobs=-1, random_state=0)
                random_forest.fit(X, y)

                # Assign the random forest regressor to each user
                self.user_profile[u] = random_forest
        if self.regressor_method == "neural network":
            feature_names = self.content_features.columns.tolist()
            for u in self.user_profile:
                inner_item_ids, user_ratings = zip(*self.trainset.ur[u])
                df_user = pd.DataFrame({"item_id": inner_item_ids, "user_ratings": user_ratings})

                # Map inner_item_id to raw_item_id
                df_user["item_id"] = df_user["item_id"].map(self.trainset.to_raw_iid)

                # Merge user ratings with content features
                df_user = df_user.merge(self.content_features, how='left', left_on='item_id', right_index=True)

                # Prepare X and y for the neural network model
                X = df_user[feature_names].values
                y = df_user['user_ratings'].values

                # Fit the neural network model make sure to make it run fast
                neural_network = MLPRegressor(hidden_layer_sizes=(100, 100), max_iter=1000)
                neural_network.fit(X, y)

                # Assign the neural network regressor to each user
                self.user_profile[u] = neural_network


    def estimate(self, u, i):
        """Scoring component used for item filtering"""
        # First, handle cases for unknown users and items
        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible('User and/or item is unkown.')

        if self.regressor_method == 'random_score':
            rd.seed()
            score = rd.uniform(0.5, 5)

        elif self.regressor_method == 'random_sample':
            rd.seed()
            score = rd.choice(self.user_profile[u])

        elif self.regressor_method == 'linear_regression':
            raw_item_id = self.trainset.to_raw_iid(i)
            item_features = self.content_features.loc[raw_item_id:raw_item_id, :].values
            score = self.user_profile[u].predict(item_features)[0]

        elif self.regressor_method == 'random_forest':
            raw_item_id = self.trainset.to_raw_iid(i)
            item_features = self.content_features.loc[raw_item_id:raw_item_id, :].values
            score = self.user_profile[u].predict(item_features)[0]

        elif self.regressor_method == 'neural network':
            raw_item_id = self.trainset.to_raw_iid(i)
            item_features = self.content_features.loc[raw_item_id:raw_item_id, :].values
            score = self.user_profile[u].predict(item_features)[0]
            # implement here the other regressor prediction methods

        return score

    def explain(self,u): # takes a user id as input and give the importance of each feature for this user store in self.user_profile_explain dict
        return self.user_profile_explain[u]

# user based collaborative filtering
class UserBased(AlgoBase):
    def __init__(self, k=4, min_k=4, sim_options={"name": "pearson", "min_support": 3}, **kwargs):
        AlgoBase.__init__(self, sim_options=sim_options, **kwargs)
        self.k = k
        self.min_k = min_k

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)
        self.ratings_matrix = self.compute_rating_matrix()
        self.similarity_matrix = self.compute_similarity_matrix()
        self.mean_ratings = np.nanmean(self.ratings_matrix, axis=1)

    def estimate(self, u, i):
        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible('User and/or item is unknown.')

        estimate = self.mean_ratings[u]

        # Create the peer group of user u for item i
        potential_neighbors = [(n, self.similarity_matrix[u, n], self.ratings_matrix[n, i])
                               for n, r in self.trainset.ir[i]
                               if not np.isnan(self.ratings_matrix[n, i]) and n != u]

        # Pick the top neighbors efficiently
        top_neighbors = heapq.nlargest(self.k, potential_neighbors, key=lambda x: x[1])

        # Compute the weighted average
        weighted_sum = 0
        sim_sum = 0
        actual_k = 0

        for neighbor, similarity, rating in top_neighbors:
            weighted_sum += similarity * (rating - self.mean_ratings[neighbor])
            sim_sum += abs(similarity)
            actual_k += 1

        # If we have enough neighbors in the peer group, add the weighted average to the user average estimation
        if actual_k >= self.min_k:
            estimate += weighted_sum / sim_sum

        return estimate


    def compute_rating_matrix(self):# -- implement here the compute_rating_matrix function --
        m, n = self.trainset.n_users, self.trainset.n_items

        # Initialize an mxn numpy array with NaN values
        rating_matrix = np.empty((m, n))
        rating_matrix[:] = np.nan

        # Iterate through users and their ratings
        for uiid in range(m):
            user_ratings = self.trainset.ur[uiid]

            # Iterate through each user's rated items
            for rating_tuple in user_ratings:
                # rating_tuple: (item_inner_id, rating)
                item_iid, rating = rating_tuple

                # Set the rating in the rating_matrix
                rating_matrix[uiid, item_iid] = rating




        return rating_matrix




    def compute_similarity_matrix(self):# -- implement here the compute_similarity_matrix function --
        # Get the number of users
        m = self.trainset.n_users

        # Preallocate the m x m similarity matrix with diagonal elements set to 1
        similarity_matrix = np.eye(m)

        # Get the minimum support value
        min_support = self.sim_options["min_support"]
        if self.sim_options.get('name') == 'msd':
            # Iterate through every pair of users
            for i in range(m):
                row_i = self.ratings_matrix[i]

                for j in range(i + 1, m):
                    row_j = self.ratings_matrix[j]

                    # Compute the support
                    support = np.sum(np.isnan(row_i - row_j))

                    # If the support is at least higher than min_support, compute the similarity
                    if support >= min_support:
                        if np.isnan(row_i).all() or np.isnan(row_j).all():
                            similarity = 0  # Set similarity to 0 if rows are empty
                        else:
                            # Compute the mean squared difference
                            msd = np.nanmean((row_i - row_j) ** 2)

                        # Compute the similarity based on Euclidean distance
                            similarity = 1 / (1 + np.sqrt(msd))

                        # Set the similarity values in the similarity matrix
                        similarity_matrix[i, j] = similarity
                        similarity_matrix[j, i] = similarity
            return similarity_matrix

        elif self.sim_options.get('name') == 'jaccard':
            for i in range(m):
                row_i = self.ratings_matrix[i]

                for j in range(i + 1, m):
                    row_j = self.ratings_matrix[j]

                    # Compute the Jaccard similarity
                    intersection = np.sum(np.isnan(row_i) & np.isnan(row_j))
                    union = np.sum(np.isnan(row_i) | np.isnan(row_j))
                    if union == 0:
                        similarity = 0
                    else:
                        similarity = intersection / union

                    # Set the similarity values in the similarity matrix
                    similarity_matrix[i, j] = similarity
                    similarity_matrix[j, i] = similarity
            return similarity_matrix

        elif self.sim_options.get('name') == 'cosine':
            for i in range(m):
                row_i = self.ratings_matrix[i]

                for j in range(i + 1, m):
                    row_j = self.ratings_matrix[j]

                    # Compute the cosine similarity
                    numerator = np.sum(row_i * row_j)
                    denominator = np.sqrt(np.sum(row_i ** 2)) * np.sqrt(np.sum(row_j ** 2))
                    if denominator == 0:
                        similarity = 0
                    else:
                        similarity = numerator / denominator

                    # Set the similarity values in the similarity matrix
                    similarity_matrix[i, j] = similarity
                    similarity_matrix[j, i] = similarity
            return similarity_matrix
        elif self.sim_options.get('name') == 'pearson':
            for i in range(m):
                row_i = self.ratings_matrix[i]

                for j in range(i + 1, m):
                    row_j = self.ratings_matrix[j]

                    # Compute the Pearson correlation coefficient
                    mean_i = np.mean(row_i)
                    mean_j = np.mean(row_j)
                    centered_i = row_i - mean_i
                    centered_j = row_j - mean_j

                    numerator = np.sum(centered_i * centered_j)
                    denominator = np.sqrt(np.sum(centered_i ** 2)) * np.sqrt(np.sum(centered_j ** 2))
                    if denominator == 0:
                        similarity = 0
                    else:
                        similarity = numerator / denominator

                    # Set the similarity values in the similarity matrix
                    similarity_matrix[i, j] = similarity
                    similarity_matrix[j, i] = similarity
            print(similarity_matrix)
            return similarity_matrix


