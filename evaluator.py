

# third parties imports
import numpy as np
import pandas as pd
# -- add new imports here --

# local imports
from configs import EvalConfig
from constants import Constant as C
from loaders import export_evaluation_report
from loaders import load_ratings
# -- add new imports here --
from surprise.model_selection import *
from surprise import accuracy, SVD,Dataset
from models import get_top_n
from collections import defaultdict





def generate_split_predictions(algo, ratings_dataset, eval_config):
    """Generate predictions on a random test set specified in eval_config"""
    # -- implement the function generate_split_predictions --
    trainset, testset = train_test_split(ratings_dataset, test_size=eval_config.test_size)




    algo.fit(trainset)
    predictions = algo.test(testset)

    return predictions


def generate_loo_top_n(algo, ratings_dataset, eval_config):
    # Unpack the values from eval_config
    top_n_value = eval_config.top_n_value
    # Create a LeaveOneOut instance with one split
    loo = LeaveOneOut(n_splits=1)

    # Split the dataset into trainset and testset
    for trainset, testset in loo.split(ratings_dataset):
        pass  # Do nothing, just unpacking the trainset and testset
    # Build the anti-testset
    anti_testset = trainset.build_anti_testset()
    # Fit the algorithm with the trainset
    algo.fit(trainset)

    # Prepare the dictionary to store the top n predictions for each user
    top_n = defaultdict(list)


    # Predict ratings for all pairs (user, items) that are NOT in the training set.
    for user_id, movie_id, _ in anti_testset:
        predicted_rating = algo.predict(user_id, movie_id).est
        top_n[user_id].append((movie_id, predicted_rating))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:top_n_value]

    # store the top-n recommendations in  a dataframe
    top_n_df = pd.DataFrame(columns=["algo","user_id", "movie_id"])
    for user_id, user_ratings in top_n.items():
        for movie_id, _ in user_ratings:
            top_n_df = top_n_df.append({"algo":algo.__class__.__name__,"user_id": user_id, "movie_id": movie_id}, ignore_index=True)
    # store the top-n recommendations in the recs folder append mode
    top_n_df.to_csv(f"{C.REC_PATH}/top_loo_recs.csv", mode="a", header=True, index=False)

    return top_n, testset


def generate_full_top_n(algo, ratings_dataset, eval_config):
    # Train the algorithm on the full dataset
    top_n_value = eval_config.top_n_value
    trainset = ratings_dataset.build_full_trainset()
    algo.fit(trainset)

    # Generate the anti-test set
    anti_testset = trainset.build_anti_testset()
    top_n = []
    # Predict the ratings for the anti-test set
    for user_id, movie_id, rating in anti_testset:
        rating_estimate = algo.predict(user_id, movie_id).est
        top_n.append((user_id, movie_id, rating, rating_estimate, "n"))

    # Get the top-N recommendations for each user
    anti_testset_top_n = get_top_n(top_n, top_n_value)

    #store the top-n recommendations in  a dataframe
    top_n_df = pd.DataFrame(columns=["algo","user_id", "movie_id"])
    for user_id, user_ratings in anti_testset_top_n.items():
        for movie_id, _ in user_ratings:
            top_n_df = top_n_df.append({"algo":algo.__class__.__name__,"user_id": user_id, "movie_id": movie_id}, ignore_index=True)
     #store the top-n recommendations in the recs folder append mode
    top_n_df.to_csv(f"{C.REC_PATH}/top_n_recs.csv", mode="a", header=True, index=False)

    return anti_testset_top_n


def precompute_information(surprise_ratings_dataset):
    # Ensure the dataset is in the trainset format
    if not hasattr(surprise_ratings_dataset, 'ur'):
        surprise_ratings_dataset = surprise_ratings_dataset.build_full_trainset()

    # Count the number of ratings for each movie
    movie_ratings_count = defaultdict(int)
    for _, movie_id, _ in surprise_ratings_dataset.all_ratings():
        movie_ratings_count[movie_id] += 1

    # Sort the movies by the number of ratings (popularity)
    sorted_movies = sorted(movie_ratings_count.items(), key=lambda x: x[1], reverse=True)

    # Create a dictionary that maps each movie to its popularity rank
    item_to_rank = {surprise_ratings_dataset.to_raw_iid(movie_id): rank + 1 for rank, (movie_id, _) in
                    enumerate(sorted_movies)}

    precomputed_dict = {"movie_id_to_rank": item_to_rank}
    return precomputed_dict


def create_evaluation_report(eval_config, sp_ratings, precomputed_dict, available_metrics):
    """ Create a DataFrame evaluating various models on metrics specified in an evaluation config.
    """
    evaluation_dict = {}
    for model_name, model, arguments in eval_config.models:
        print(f'Handling model {model_name}')
        algo = model(**arguments)
        # print the column of trainset note it is in surprise format
        if model_name == "SVD":  # perform a grid search to find the best parameters
            # create a parameter grid with a lot of parameters
            param_grid = {"n_epochs": [x for x in range(10,21)], "lr_all": [0.016,0.018,0.02,0.022,0.24,0.26,0.028,0.03], "reg_all": [0.1,0.15,0.2,0.25,0.3]}
            gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3)
            print(2)
            gs.fit(sp_ratings)
            algo = gs.best_estimator['rmse']
            # print (ge value of the best parameters
            print(gs.best_params)
        evaluation_dict[model_name] = {}

        # Type 1 : split evaluations
        if len(eval_config.split_metrics) > 0:
            print('Training split predictions')
            predictions = generate_split_predictions(algo, sp_ratings, eval_config)
            for metric in eval_config.split_metrics:
                print(f'- computing metric {metric}')
                assert metric in available_metrics['split']
                evaluation_function, parameters = available_metrics["split"][metric]
                evaluation_dict[model_name][metric] = evaluation_function(predictions, **parameters)
                print(f'-- metric {metric} : {evaluation_dict[model_name][metric]}')

                # Type 2 : loo evaluations
        if len(eval_config.loo_metrics) > 0:
            print('Training loo predictions')
            anti_testset_top_n, testset = generate_loo_top_n(algo, sp_ratings, eval_config)
            for metric in eval_config.loo_metrics:
                assert metric in available_metrics['loo']
                evaluation_function, parameters = available_metrics["loo"][metric]
                evaluation_dict[model_name][metric] = evaluation_function(anti_testset_top_n, testset, **parameters)
        # Type 3 : full evaluations
        if len(eval_config.full_metrics) > 0:
            print('Training full predictions')
            anti_testset_top_n = generate_full_top_n(algo, sp_ratings, eval_config)
            for metric in eval_config.full_metrics:
                assert metric in available_metrics['full']
                evaluation_function, parameters = available_metrics["full"][metric]
                evaluation_dict[model_name][metric] = evaluation_function(
                    anti_testset_top_n,
                    **precomputed_dict,
                    **parameters
                )
        print(evaluation_dict)

    return pd.DataFrame.from_dict(evaluation_dict).T


def get_hit_rate(anti_testset_top_n, testset):
    """Compute the average hit over the users (loo metric)

    A hit (1) happens when the movie in the testset has been picked by the top-n recommender
    A fail (0) happens when the movie in the testset has not been picked by the top-n recommender
    """

    # Convert testset list to a dictionary
    testset_dict = {}
    for user, item, _ in testset:
        testset_dict[user] = item

    hits = 0
    total_users = len(anti_testset_top_n)

    for user in anti_testset_top_n:
        # check if the movie in the testset has been picked by the top-n recommender
        if testset_dict[user] in [movie_id for (movie_id, _) in anti_testset_top_n[user]]:
            hits += 1

    hit_rate = hits / total_users if total_users > 0 else 0
    return hit_rate


def get_novelty(anti_testset_top_n, movie_id_to_rank): #calculate the average novelty over the users (full metric)
    """Compute the average novelty over the users (full metric)"""


    # Compute the novelty for each user
    novelties = []
    for user in anti_testset_top_n:
        novelty = 0
        # loop through the top-n recommendations for each user
        for movie_id, rating_estimate in anti_testset_top_n[user]:
            novelty += movie_id_to_rank[movie_id]
        novelty /= len(anti_testset_top_n[user])
        novelties.append(novelty)

    # Compute the average novelty over all users
    novelty = sum(novelties) / len(novelties) if len(novelties) > 0 else 0
    # normalize the novelty in [0,1]
    if novelty >0:
        novelty = (novelty - 1) / (max(movie_id_to_rank.values()) - 1)
    return novelty


def precision_loo(anti_testset_top_n, testset):
    # Convert testset to a dictionary for easier look-up.
    testset_dict = {user: item for user, item, _ in testset}
    true_positives = 0
    total_predicted_positives = 0
    # Convert anti_testset_top_n list to a dictionary


    for user in anti_testset_top_n:
        if user not in testset_dict:
            continue
        user_testset = {testset_dict[user]}
        # add to true the number of movies in the testset that are in the top-n recommendations
        true_positives += len(user_testset.intersection(set([movie_id for (movie_id, _) in anti_testset_top_n[user]])))
        total_predicted_positives += len(anti_testset_top_n[user])

    precision = true_positives / total_predicted_positives if total_predicted_positives > 0 else 0

    return precision


AVAILABLE_METRICS = {
    "split": {
        "mae": (accuracy.mae, {'verbose': False}),
        "rmse": (accuracy.rmse, {'verbose': False}),
    },
    "loo": {
        "hit rate": (get_hit_rate,{}),
        "precision":(precision_loo,{}),
    },
    "full": {
        "novelty": (get_novelty,{}),
    }
    # -- add new types of metrics here --
}

sp_ratings = load_ratings(surprise_format=True)
precomputed_dict = precompute_information(sp_ratings)

evaluation_report = create_evaluation_report(EvalConfig, sp_ratings, precomputed_dict, AVAILABLE_METRICS)
print(evaluation_report)
export_evaluation_report(evaluation_report)
