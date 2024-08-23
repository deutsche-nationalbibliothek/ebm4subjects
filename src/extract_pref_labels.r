#!/usr/bin/env Rscript
library("optparse")

option_list = list(
  make_option(
    c("-g", "--gnd-ttl"), type = "character", default = "vocab/gnd.ttl",
    help = "Path to the GND prefLabel RDF file (default: vocab/gnd.ttl)"
  ),
  make_option(
    c("-o", "--output"), type = "character", default = "vocab/gnd-pref-labels.arrow",
    help = "Path to the output CSV file (default: gnd-pref-labels.csv)"
  )
)

opt_parser <- OptionParser(option_list = option_list)
opt <- parse_args(opt_parser)

message("Loading gnd-pref-labels with rdftab...")
call_to_rdftab <- paste0("rdftab --no-predicate --no-language \\
    --strip-base-uri \"http://d-nb.info/gnd/\" \\
    -p \"http://www.w3.org/2004/02/skos/core#prefLabel\" \\
    ", opt$`gnd-ttl`)

gnd <- readr::read_csv(
  pipe(call_to_rdftab), skip = 1,
  show_col_types = FALSE,
  col_types = "cc", col_names = c("label_id", "label_text")
)

message("Writing gnd-pref-labels to", opt$output)
arrow::write_feather(gnd, opt$output)
