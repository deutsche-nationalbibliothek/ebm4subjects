import argparse
import pickle

import pandas as pd
import xgboost as xgb

from prepare_data_for_gbm import prepare_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process command line arguments")
    parser.add_argument(
        "--model_file",
        type=str,
        default="results/train/model.pkl",
        help="Path to the trained model file",
    )
    parser.add_argument(
        "--test_index",
        type=str,
        default="corpora/ft30k/test.arrow",
        help="Path to the test index file",
    )
    parser.add_argument(
        "--test_candidates",
        type=str,
        default="results/train/candidates.arrow",
        help="Path to the predictions file",
    )
    parser.add_argument(
        "--max_docs",
        type=int,
        default=-1,
        help="Maximum number of documents to use",
    )
    parser.add_argument(
        "--label_distribution",
        type=str,
        default="assets/gnd-label-distribution.arrow",
        help="Path to the label distribution file",
    )
    parser.add_argument(
        "--top_k", type=int, default=100, help="Number of top candidates to consider"
    )
    parser.add_argument(
        "--out_file",
        type=str,
        default="results/test/predictions_new.arrow",
        help="Path to the output file",
    )
    args = parser.parse_args()

    model = pickle.load(open(args.model_file, "rb"))

    print("loading data...")
    test_index = pd.read_feather(args.test_index)
    test_candidates = pd.read_feather(args.test_candidates)
    gnd_label_distribution = pd.read_feather(args.label_distribution).filter(
        items=["label_id", "label_freq"]
    )

    print("preparing data...")
    model_data = prepare_data(
        index=test_index,
        include_ground_truth=False,
        candidates=test_candidates,
        ground_truth=None,
        label_distribution=gnd_label_distribution,
        max_docs=args.max_docs,
    )

    xgb_matrix = model_data.filter(
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
    dtest = xgb.DMatrix(xgb_matrix)

    print("making predictions...")
    predictions = model.predict(dtest)

    model_data["score"] = predictions
    model_data = (
        model_data.filter(items=["doc_id", "label_id", "score"])
        .sort_values(["doc_id", "score"], ascending=False)
        .groupby("doc_id")
        .head(args.top_k)
    )

    model_data.to_feather(args.out_file)
    
    # print(model_data)
    # python src/predict.py --model_file results/train/model.pkl --test_index corpora/ftoa/test.arrow --test_candidates results/test/candidates.arrow --max_docs 30000 --label_distribution assets/gnd-label-distribution.arrow --top_k 100 --out_file results/test/predictions_new.arrow
