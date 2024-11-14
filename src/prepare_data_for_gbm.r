require(dplyr)
require(stringr)
require(aeneval)

prepare_data <- function(
  index,
  candidates,
  label_disribution,
  include_ground_truth = FALSE,
  ground_truth = NULL,
  max_docs = -1
) {
  if (max_docs > 0) {
    message("Using only ", max_docs, " documents.")
    index <- index  |>
      slice_head(n = max_docs)
  }

  if (include_ground_truth) {

    gold_standard <- index %>%
      inner_join(gt, by = c("idn", "kind"), na_matches = "never")  |>
      mutate(
        doc_id = idn,
        label_id = str_match(uri, "(?<idn>[0-9X]{9,10})")[, "idn"]
      )

    n_idn_not_predicted <- anti_join(
      distinct(gold_standard, doc_id),
      distinct(candidates, doc_id),
      by = "doc_id"
    )  |> nrow()

    if (n_idn_not_predicted > 0) {
      message(
        "There are ",
        n_idn_not_predicted,
        " documents in the ground truth with no candidates."
      )
    }

    res <- create_comparison(
      gold_standard,
      candidates
    )  |>
      left_join(label_disribution, by = "label_id", na_matches = "never")

    res <- res  |>
      filter(suggested)  |>
      mutate(gold = ifelse(gold, 1.0, 0.0))  |>
      select(
        gold,
        score,
        label_freq,
        occurrences,
        first_occurence,
        last_occurence,
        spread,
        is_prefLabel
      )
  } else {
    index <- select(index, doc_id = idn)

    n_idn_not_predicted <- anti_join(
      index,
      distinct(candidates, doc_id),
      by = "doc_id"
    )  |> nrow()

    if (n_idn_not_predicted > 0) {
      message(
        "There are ",
        n_idn_not_predicted,
        " documents in the index with no candidates."
      )
    }

    res <- index  |>
      inner_join(candidates, by = "doc_id", na_matches = "never")  |>
      left_join(label_disribution, by = "label_id", na_matches = "never")  |>
      select(
        doc_id,
        label_id,
        score,
        label_freq,
        occurrences,
        first_occurence,
        last_occurence,
        spread,
        is_prefLabel
      )
  }
}
