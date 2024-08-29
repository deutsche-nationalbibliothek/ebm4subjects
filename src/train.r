#!/usr/bin/env Rscript
library("optparse")

option_list = list(
  # make_option(
  #   c("--kind"), type = "character", default = "title",
  #   help = "title or ft",
  #   metavar = "character"
  # ),
  # make_option(
  #   c("--task"), type = "character", default = "title",
  #   help = "name of the task e.g. ft30k or titleoa",
  #   metavar = "character"
  # ),
  # make_option(
  #   c("--evalset"), type = "character", default = "test",
  #   help = "which split to evaluate on, e.g. test or validate",
  #   metavar = "character"
  # ),
  make_option(
    c("--ground_truth"),
    type = "character", default = "corpora/ground-truth.arrow",
    help = "path to the ground truth file",
    metavar = "character"
  ),
  make_option(
    c("--train_index"),
    type = "character", default = "corpora/ftoa/train.arrow",
    help = "path to the test index file",
    metavar = "character"
  ),
  make_option(
    c("--train_candidates"),
    type = "character", default = "results/train/candidates.arrow",
    help = "path to the predictions file",
    metavar = "character"
  ),
  make_option(
    c("--label_distribution"), type = "character",
    default = "assets/gnd-label-distribution.arrow",
    help = "path to the label distribution file",
    metavar = "character"
  ),
  make_option(
    c("--model_file"),
    type = "character", default = "results/train/model.rds",
    help = "path to the output file",
    metavar = "character"
  )
)

opt_parser <- OptionParser(option_list = option_list)
opt <- parse_args(opt_parser)

suppressPackageStartupMessages(library(tidyverse))
suppressPackageStartupMessages(library(future))
suppressPackageStartupMessages(library(arrow))
library(aeneval)
library(tree)
library(gbm)

source("src/prepare_data_for_gbm.r")

message("loading data...")
gt <- read_feather(opt$ground_truth)
gnd_label_disribution <- read_feather(
  opt$label_distribution,
  col_select = c("label_id", "label_freq")
)

index_train <- read_feather(opt$train_index)
candidates_train <- read_feather(opt$train_candidates)

index_validate <- read_feather("corpora/ftoa/validate.arrow")
candidates_validate <- read_feather("results/validate/candidates.arrow")

message("preparing data...")
model_data_train <- prepare_data(
  index = index_train,
  include_ground_truth = TRUE,
  ground_truth = gt,
  candidates = candidates_train,
  label_disribution = gnd_label_disribution
)

# tree_model <- tree(
#   gold ~ .,
#   data = model_data_train
# )

# plot(tree_model)
# text(tree_model, pretty = 0)
message("training model...")
bst <- gbm(
  gold ~ .,
  data = model_data_train,
  distribution = "bernoulli",
  n.trees = 100,
  interaction.depth = 4,
  shrinkage = 0.2,
  verbose = TRUE
)

message("saving model...")
saveRDS(bst, opt$model_file)

# model_data_validate <- prepare_data(
#   index_validate,
#   gt,
#   candidates_validate,
#   gnd_label_disribution)

# #preds <- predict(tree_model, model_data_validate, type = "class")
# preds <- predict(bst, model_data_validate, type = "response")

# library(pROC)

# roc_auc <- roc(model_data_validate$gold, preds)$auc
# roc_auc

