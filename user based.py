# third parties imports
import numpy as np
import pandas as pd
from surprise import AlgoBase
# -- add new imports here --
import heapq
# local imports
from constants import Constant as C
from loaders import load_ratings
from configs import EvalConfig
# -- add new imports here --
from surprise.model_selection import *
from surprise.accuracy import *
from surprise import Reader
from surprise import Dataset,KNNWithMeans
from surprise import accuracy, SVD
from surprise.similarities import msd
# -- load data, build trainset and anti testset --
ratings = load_ratings(True)
trainset = ratings.build_full_trainset()

# -- using surprise's user-based algorithm, explore the impact of different parameters and displays predictions -

sim_options = {"user_based": True, "min_support": 3}
algo = KNNWithMeans(k = 3,min_k=2,sim_options=sim_options)
algo.fit(trainset)
print(algo.predict(11,364))


class UserBased(AlgoBase):
    def __init__(self, k=3, min_k=2, sim_options={"name": "msd", "min_support": 3}, **kwargs):
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




# print the user and item ids of the element of the surprise trainset
algo2 = UserBased(sim_options={'name': 'msd', 'min_support': 3})
algo2.fit(trainset)
print(algo2.predict(11,364))