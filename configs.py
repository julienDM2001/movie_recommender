# local imports
from models import *


class EvalConfig:
    models = [
        ("Based on what you like", ContentBased, {}),
        ("Based on similar users", UserBased, {}),
        ("Based on magical mathematics", SVD, {"n_factors": 70, "n_epochs": 10, "lr_all": 0.016, "reg_all": 0.3}),
    ]

    split_metrics = ["mae", "rmse"]
    loo_metrics = ["hit rate", "precision"]
    full_metrics = ["novelty"]

    # Split parameters
    test_size = 0.3  # -- configure the test_size (from 0 to 1) --

    # Loo parameters
    top_n_value = 40  # -- configure the number of recommendations (> 1) --
