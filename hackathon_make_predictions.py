import pandas as pd

from constants import Constant as C
from loaders import load_ratings
from models import ContentBased


def make_hackathon_prediction(feature_method, regressor_method):
    """Generate a prediction file on the test set"""
    # 1) load train data - make sure to redirect the DATA_PATH to'data/hackathon'
    assert str(C.DATA_PATH) == 'hackathon'
    sp_ratings = load_ratings(surprise_format=True)
    train_set = sp_ratings.build_full_trainset()

    # 2) train your ContentBased model on the train set
    content_knn = ContentBased(feature_method, regressor_method)
    content_knn.fit(train_set)

    # 3) make predictions on the test set
    df_test = pd.read_csv('hackathon/evidence/ratings_test.csv')[C.USER_ITEM_RATINGS]
    test_records = list(df_test.to_records(index=False))
    predictions = content_knn.test(test_records)
    output_predictions = []
    for uid, iid, _, est, _ in predictions:
        output_predictions.append([uid, iid, est])
    df_predictions = pd.DataFrame(data=output_predictions, columns=df_test.columns)

    # 4) dump predictions
    df_predictions.to_csv(f'ratings_predictions.csv', index=False)


make_hackathon_prediction(None, "random_score")