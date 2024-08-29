#!/usr/bin/env Rscript
library("optparse")

# Define the list of options
option_list = list(
  make_option(
    c("--model_file"),
    type = "character", default = "results/train/model.rds",
    help = "path to the trained model file",
    metavar = "character"
  ),
  make_option(
    c("--test_index"),
    type = "character", default = "corpora/ftoa/test.arrow",
    help = "path to the test index file",
    metavar = "character"
  ),
  make_option(
    c("--test_candidates"),
    type = "character", default = "results/test/candidates.arrow",
    help = "path to the test candidates file",
    metavar = "character"
  ),
  make_option(
    c("--label_distribution"), type = "character",
    default = "assets/gnd-label-distribution.arrow",
    help = "path to the label distribution file",
    metavar = "character"
  ),
  make_option(
    c("--top_k"), type = "integer",
    default = 100L,
    help = "number of top candidates to consider",
    metavar = "integer"
  ),
  make_option(
    c("--out_file"),
    type = "character", default = "results/test/predictions.arrow",
    help = "path to the output predictions file",
    metavar = "character"
  )
)

# Parse the options
opt_parser <- OptionParser(option_list = option_list)
opt <- parse_args(opt_parser)

# Load necessary libraries
suppressPackageStartupMessages(library(tidyverse))
suppressPackageStartupMessages(library(future))
suppressPackageStartupMessages(library(arrow))
library(aeneval)
library(gbm)

source("src/prepare_data_for_gbm.r")

# Load the model
bst <- readRDS(opt$model_file)

# Load the test data
message("loading data...")
test_index <- read_feather(opt$test_index)
test_candidates <- read_feather(opt$test_candidates)
gnd_label_disribution <- read_feather(
  opt$label_distribution,
  col_select = c("label_id", "label_freq")
)

# Prepare the data for prediction
message("preparing data...")
model_data <- prepare_data(
  index = test_index,
  include_ground_truth = FALSE,
  candidates = test_candidates,
  label_disribution = gnd_label_disribution
)

# Make predictions
message("making predictions...")
preds <- predict(bst, model_data, type = "response")

predictions <- model_data  |>
  mutate(score = preds)  |>
  select(doc_id, label_id, score)  |>
  group_by(doc_id)  |>
  arrange(desc(score))  |>
  slice_head(n = opt$top_k)  |>
  ungroup()

write_feather(predictions, opt$out_file)