import argparse
import pickle

import pandas as pd
import xgboost as xgb

from prepare_data_for_gbm import prepare_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process command line arguments")
    parser.add_argument(
        "--ground_truth",
        type=str,
        default="corpora/ground-truth.arrow",
        help="Path to the ground truth file",
    )
    parser.add_argument(
        "--train_index",
        type=str,
        default="corpora/ftoa/train.arrow",
        help="Path to the test index file",
    )
    parser.add_argument(
        "--max_docs",
        type=int,
        default=-1,
        help="Maximum number of documents to use",
    )
    parser.add_argument(
        "--train_candidates",
        type=str,
        default="results/train/candidates.arrow",
        help="Path to the predictions file",
    )
    parser.add_argument(
        "--label_distribution",
        type=str,
        default="assets/gnd-label-distribution.arrow",
        help="Path to the label distribution file",
    )
    parser.add_argument(
        "--n_rounds", type=int, default=100, help="Number of boosting iterations"
    )
    parser.add_argument(
        "--interaction_depth",
        type=int,
        default=4,
        help="Maximum depth of interaction in the GBM model",
    )
    parser.add_argument(
        "--shrinkage",
        type=float,
        default=0.2,
        help="Shrinkage parameter for the GBM model between 0 and 1",
    )
    parser.add_argument(
        "--subsample",
        type=float,
        default=0.5,
        help="Subsample ratio of the training instances",
    )
    parser.add_argument(
        "--verbose",
        type=bool,
        default=False,
        help="Whether to print verbose output during training",
    )
    parser.add_argument(
        "--model_file",
        type=str,
        default="results/train/model.pkl",
        help="Path to the output file",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=20,
        help="Number of jobs to run in parallel",
    )
    args = parser.parse_args()

    print("loading data...")
    ground_truth = pd.read_feather(args.ground_truth)
    gnd_label_distribution = pd.read_feather(
        args.label_distribution, columns=["label_id", "label_freq"]
    )
    index_train = pd.read_feather(args.train_index)
    candidates_train = pd.read_feather(args.train_candidates)

    print("preparing data...")
    model_data_train = prepare_data(
        index=index_train,
        include_ground_truth=True,
        ground_truth=ground_truth,
        candidates=candidates_train,
        label_distribution=gnd_label_distribution,
        max_docs=args.max_docs,
    )

    print("training model...")

    xgb_matrix = model_data_train.filter(
        items=[
            "score",
            "label_freq",
            "occurrences",
            "first_occurence",
            "last_occurence",
            "spread",
            "is_prefLabel",
        ]
    )

    dtrain = xgb.DMatrix(xgb_matrix, model_data_train["gold"])
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "eta": args.shrinkage,
        "max_depth": args.interaction_depth,
        "subsample": args.subsample,
        "verbosity": 1,
        "nthread": args.n_jobs,
    }
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=args.n_rounds,
        evals=[(dtrain, "train")],
    )

    print("saving model...")
    pickle.dump(model, open(args.model_file, "wb"))

# python src/train.py --ground_truth corpora/ground-truth.arrow --train_index corpora/ftoa/train.arrow --max_docs 30000 --train_candidates results/train/candidates.arrow --label_distribution assets/gnd-label-distribution.arrow --model_file results/train/model.pkl --n_rounds 812 --subsample 0.62 --interaction_depth 7 --shrinkage 0.023 --verbose true --n_jobs 10
