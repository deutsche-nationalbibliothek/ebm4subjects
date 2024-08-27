library(aeneval)
library(tidyverse)
library(arrow)

predictions <- arrow::read_feather("results/test/predictions.arrow")  |>
  group_by(doc_id)  |>
  mutate(rank = min_rank(-score))  |>
  ungroup()

index <- arrow::read_feather("corpora/ft30k/test.arrow")
gt <- arrow::read_feather("corpora/ground-truth.arrow")

label_distribution <- arrow::read_feather("assets/gnd-label-distribution.arrow")

gold_standard <- index  |>
  select(idn, kind)  |>
  left_join(gt, by = c("idn", "kind"))  |>
  transmute(
    doc_id = idn,
    label_id = str_match(uri, "(?<idn>[0-9X]{9,10})")[,"idn"]
  )

pref_labels <- arrow::read_feather("vocab/gnd_pref_labels.arrow")

compute_set_retrieval_scores(
  gold_standard = gold_standard,
  predicted = filter(predictions, rank <= 5)
)

comp <- create_comparison(
  gold_standard = gold_standard,
  predicted = filter(predictions, rank <= 10),
)  |>
  left_join(
    pref_labels,
    by = c("label_id")
  )  |>
  arrange(doc_id)  |>
  select(doc_id, gold, suggested, label_text, score, occurrences, rank)
