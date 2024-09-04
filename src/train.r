#!/usr/bin/env Rscript
library("optparse")

option_list = list(
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
    c("--max_docs"),
    type = "integer", default = -1,
    help = "maximum number of documents to use",
    metavar = "integer"
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
    c("--n_trees"),
    type = "integer", default = 100,
    help = "number of trees in the GBM model",
    metavar = "integer"
  ),
  make_option(
    c("--interaction_depth"),
    type = "integer", default = 4,
    help = "maximum depth of interaction in the GBM model",
    metavar = "integer"
  ),
  make_option(
    c("--shrinkage"),
    type = "numeric", default = 0.2,
    help = "shrinkage parameter for the GBM model",
    metavar = "numeric"
  ),
  make_option(
    c("--verbose"),
    type = "logical", default = FALSE,
    help = "whether to print verbose output during training",
    metavar = "logical"
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

message("preparing data...")
model_data_train <- prepare_data(
  index = index_train,
  include_ground_truth = TRUE,
  ground_truth = gt,
  candidates = candidates_train,
  label_disribution = gnd_label_disribution,
  max_docs = opt$max_docs
)

# tree_model <- tree(
#   gold ~ .,
#   data = model_data_train
# )

# plot(tree_model)
# text(tree_model, pretty = 0)
message("training model with parameters:\n n_trees = ", opt$n_trees,
        ",\n interaction_depth = ", opt$interaction_depth,
        ",\n shrinkage = ", opt$shrinkage,
        ",\n verbose = ", opt$verbose)
bst <- gbm(
  gold ~ .,
  data = model_data_train,
  distribution = "bernoulli",
  n.trees = opt$n_trees,
  interaction.depth = opt$interaction_depth,
  shrinkage = opt$shrinkage,
  verbose = opt$verbose
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

