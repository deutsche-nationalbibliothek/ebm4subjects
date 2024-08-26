#!/usr/bin/env Rscript
library("optparse")

option_list = list(
  make_option(
    c("--kind"), type = "character", default = "title",
    help = "title or ft",
    metavar = "character"
  ),
  make_option(
    c("--task"), type = "character", default = "title",
    help = "name of the task e.g. ft30k or titleoa",
    metavar = "character"
  ),
  make_option(
    c("--evalset"), type = "character", default = "test",
    help = "which split to evaluate on, e.g. test or validate",
    metavar = "character"
  ),
  make_option(
    c("--ground_truth"),
    type = "character", default = "corpora/ground-truth.arrow",
    help = "path to the ground truth file",
    metavar = "character"
  ),
  make_option(
    c("--test_index"),
    type = "character", default = "corpora/title/test.arrow",
    help = "path to the test index file",
    metavar = "character"
  ),
  make_option(
    c("--predictions_file"),
    type = "character", default = "results/test/predictions.arrow",
    help = "path to the predictions file",
    metavar = "character"
  ),
  make_option(
    c("--out_folder"),
    type = "character", default = "results/test",
    help = "path to the output folder",
    metavar = "character"
  ),
  make_option( 
    c("--n_jobs"), type = "integer", default = 20,
    help = "number of jobs to run in parallel",
    metavar = "integer"
  )
)

opt_parser <- OptionParser(option_list = option_list)
opt <- parse_args(opt_parser)

suppressPackageStartupMessages(library(tidyverse))
suppressPackageStartupMessages(library(future))
suppressPackageStartupMessages(library(arrow))
library(aeneval)

message("Switching to multicore exectution with ", opt$n_jobs, " workers.")
plan(multicore, workers = opt$n_jobs)

# Lese Vorschläge von Attention-XML Test
message("loading predictions...")
predicted <- arrow::read_feather(opt$predictions_file)
predicted <- predicted  |>
  group_by(doc_id)  |>
  mutate(rank = min_rank(-score))  |>
  ungroup()

# Lese Gold-Standard für gesamten Korpus
message("loading ground truth...")
gold_standard <- arrow::read_feather(opt$ground_truth) |>
    filter(kind == opt$kind) |>
    rename(label_id = uri) |>
    mutate(label_id = str_match(label_id, "(?<idn>[0-9X]{9,10})")[,"idn"])

test_corpus <- read_feather(opt$test_index)

# Schränke Gold-Standard auf gleiche Dokumenten-Menge ein, wie predicted
gold_standard <- gold_standard |>
  inner_join(test_corpus, by = c("idn", "kind")) |>
  rename(doc_id = idn)

predictions_at_5 <- filter(predicted, rank <= 5)

# Berechne die Retrieval Metriken "at 5"
# mit Konfidenzintervallen
message("compute set retrieval scores at 5...")
res_at_5 <- aeneval::compute_set_retrieval_scores(
  gold_standard,
  predictions_at_5
)

# bringe die Ergebnisse in ein für "dvc metrics" günstiges Format
print_metric_to_json <- function(metric_name) {
  res_at_5 |>
    transmute(metric, value) |>
    filter(metric == metric_name) |>
    pivot_wider(
      names_from = metric,
      values_from = c(value),
      names_glue = "{metric}_at5_{.value}"
    ) |>
    rjson::toJSON() |>
    write_lines(
      file = file.path(opt$out_folder, paste0(metric_name, "_at_5.json"))
    )
}

json_output <- res_at_5$metric |>
  purrr::map(.f = print_metric_to_json)

# Berechne die Precision-Recall-Kurve
message("compute pr-curve...")
pr_curve <- compute_pr_curve(
  gold_standard,
  predicted,
  .verbose = TRUE,
  .progress = TRUE
)

write_csv(select(pr_curve$plot_data,
                 recall = rec,
                 precision = prec_cummax),
          file.path(opt$out_folder, "pr_curve.csv"))

# Berechne den Precision Recall AUC mit Konfidenzintervallen
message("compute pr-auc...")
pr_auc <- compute_pr_auc_from_curve(
  pr_curve
)

json_output_pr_auc <- pr_auc |>
  transmute(pr_auc) |>
  rjson::toJSON() |>
  write_lines(file = file.path(opt$out_folder, "pr_auc.json"))

message("generating plot")
g <- ggplot(pr_curve$plot_data, aes(x = rec, y = prec_cummax)) +
  geom_point() +
  geom_line() +
  ggtitle(paste0("Precision-Recall-Kurve", opt$opt$evalset, "-Set"),
          paste("prAUC =", round(pr_auc$pr_auc[1], 3))) +
  coord_fixed(xlim = c(0,1)) +
  xlab("Recall") +
  ylab("Precision")

ggsave(filename =  file.path(opt$out_folder, "pr_curve_plot.svg"),
       g, device = "svg")

# pairwise_comparison <- create_comparison(gold_standard, predictions_at_5)
# write_feather(pairwise_comparison, sink = file.path("intermediate-results", opt$opt$evalset, "pairwise_comparison.arrow"))
# results_doc_wise <- compute_intermediate_results(pairwise_comparison, grouping_var = "doc_id")
# write_feather(results_doc_wise$results_table, sink = file.path("intermediate-results", opt$opt$evalset, "results_doc_wise.arrow"))
