import pandas as pd
import random as rd
from surprise import AlgoBase
from surprise.prediction_algorithms.predictions import PredictionImpossible
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from loaders import *
from constants import Constant as C
import os
df_items = load_items()
df_ratings = load_ratings()

# Example 1 : create title_length features
df_features = df_items[C.LABEL_COL].apply(lambda x: len(x)).to_frame('n_character_title')


class ContentBased(AlgoBase):
    def __init__(self, features_method, regressor_method):
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
            df_features = df_features.drop(columns=[C.LABEL_COL])
        else:  # (implement other feature creations here)
            raise NotImplementedError(f'Feature method {features_method} not yet implemented')
        return df_features

    def fit(self, trainset):
        """Profile Learner"""
        AlgoBase.fit(self, trainset)

        # Preallocate user profiles
        self.user_profile = {u: None for u in trainset.all_users()}

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

                # Fit the random forest model
                random_forest = RandomForestRegressor(n_estimators=100)
                random_forest.fit(X, y)

                # Assign the random forest regressor to each user
                self.user_profile[u] = random_forest

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

        else:
            score = None
            # implement here the other regressor prediction methods

        return score
def test_contentbased_class(feature_method, regressor_method):
    """Test the ContentBased class.
    Tries to make a prediction on the first (user,item ) tuple of the anti_test_set
    """
    sp_ratings = load_ratings(surprise_format=True)
    train_set = sp_ratings.build_full_trainset()
    content_algo = ContentBased(feature_method, regressor_method)
    content_algo.fit(train_set)
    anti_test_set_first = train_set.build_anti_testset()[0]
    prediction = content_algo.predict(anti_test_set_first[0], anti_test_set_first[1])
    print(prediction)
