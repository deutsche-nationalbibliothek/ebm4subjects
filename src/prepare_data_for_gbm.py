import pandas as pd

pd.set_option("future.no_silent_downcasting", True)


def create_comparison(
    gold_standard: pd.DataFrame, predictions: pd.DataFrame
) -> pd.DataFrame:
    outer = gold_standard.merge(predictions, on=["doc_id"], how="outer", indicator=True)
    gold_wo_predicted = outer[outer._merge == "left_only"].drop("_merge", axis=1)
    predicted_wo_gold = outer[outer._merge == "right_only"].drop("_merge", axis=1)

    if len(predicted_wo_gold.index) != 0:
        return
    if len(gold_wo_predicted.index) > 0:
        print("Gold standard data contains documents that are not in predicted set")

    gold_standard["gold"] = True
    predictions["suggested"] = True
    compare = pd.merge(
        left=gold_standard,
        right=predictions,
        how="outer",
        on=["doc_id", "label_id"],
    ).fillna(value={"gold": False, "suggested": False})

    return compare


def prepare_data(
    index: pd.DataFrame,
    include_ground_truth: bool,
    ground_truth: pd.DataFrame | None,
    candidates: pd.DataFrame,
    label_distribution: pd.DataFrame,
    max_docs: int,
) -> pd.DataFrame:
    if max_docs > 0:
        print(f"Using only {max_docs} documents.")
        index = index.head(max_docs)

    if include_ground_truth:
        gold_standard = pd.merge(
            left=index, right=ground_truth, on=["idn", "kind"], how="inner"
        )
        gold_standard["doc_id"] = gold_standard["idn"]
        gold_standard["label_id"] = gold_standard["uri"].str.extract(r"([0-9X]{9,10})")

        outer = pd.merge(
            left=gold_standard[["doc_id"]],
            right=candidates[["doc_id"]],
            how="outer",
            indicator=True,
        )
        n_idn_not_predicted = len(
            outer[outer._merge == "left_only"].drop("_merge", axis=1).index
        )

        if n_idn_not_predicted > 0:
            print(
                f"There are {n_idn_not_predicted} documents in the ground truth with no candidates."
            )
        result = create_comparison(gold_standard, candidates)
        result = pd.merge(
            left=result, right=label_distribution, on=["label_id"], how="left"
        )

        result = result.loc[result["suggested"]]
        result["gold"] = [1 if i else 0 for i in result["gold"].tolist()]
        result = result.filter(
            items=[
                "gold",
                "score",
                "label_freq",
                "occurrences",
                "first_occurence",
                "last_occurence",
                "spread",
                "is_prefLabel",
            ]
        )
    else:
        index = index.filter(items=["idn"]).rename(columns={"idn": "doc_id"})

        outer = pd.merge(
            left=index, right=candidates[["doc_id"]], how="outer", indicator=True
        )
        n_idn_not_predicted = len(
            outer[outer._merge == "left_only"].drop("_merge", axis=1).index
        )

        if n_idn_not_predicted > 0:
            print(
                f"There are {n_idn_not_predicted} documents in the ground truth with no candidates."
            )

        result = pd.merge(left=index, right=candidates, on=["doc_id"], how="left")
        result = pd.merge(
            left=result, right=label_distribution, on=["label_id"], how="left"
        )
        result = result.filter(
            items=[
                "doc_id",
                "label_id",
                "score",
                "label_freq",
                "occurrences",
                "first_occurence",
                "last_occurence",
                "spread",
                "is_prefLabel",
            ]
        )

    result["is_prefLabel"] = result["is_prefLabel"].astype(bool)
    return result
