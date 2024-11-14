from generate_embeddings import generate_embeddings
import pandas as pd
import argparse
import numpy as np


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Embed chunks of text.')
  parser.add_argument('--chunks', type=str, help='Path the chunks file')
  parser.add_argument('--batch_size', type=int, help='Batch size for embeddings generation')
  parser.add_argument('--out', type=str, help='Path to embeddings output file')
  args = parser.parse_args()

  # Load chunk data
  with open(args.chunks, 'r', encoding='utf-8') as file:
     chunks = file.readlines()

  # Generate embeddings
  embeddings = generate_embeddings(chunks, args.batch_size)
  # save embeddings
  np.save(args.out, embeddings)