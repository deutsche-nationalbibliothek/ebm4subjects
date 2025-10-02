# Embedding Based Matching for Automated Subject Indexing

**NOTE: Work in progress. This repository is still under construction.**

This repository implements an algorithm for matching subjects with
sentence transformer embeddings. While all functionality of this code
can be run independently, this repository is not intended as 
standalone software, but is designed to work as a backend for the
[Annif toolkit}(https://annif.org/).

The idea of embedding based matching (EBM) is an inverted retrieval logic: 
Your target vocabulary is vectorized with a sentence transformer model, 
the embeddings are stored in a vector storage, enabling fast search across these
embeddings with the Hierarchical Navigable Small World Algorithm.
This enables fast semantic (embedding based) search across the 
vocaublary, even for extrem large vocabularies with many synonyms.

An input text to be indexed with terms from this vocabulary is embedded with the same
sentence transformer model, and sent as a query to the vector storage, resulting in
subject candidates with embeddings that are close to the query. 
Longer input texts can be chunked, resulting in multiple queries. 

Finally, a ranker model is trained, that reranks the subject candidates, using some 
numerical features collected during the matching process. 

This design borrows a lot of ideas from lexical matching like Maui [1], Kea [2] and particularly
[Annifs](https://annif.org/) implementation in the [MLLM-Backend](https://github.com/NatLibFi/Annif/wiki/Backend%3A-MLLM) (Maui Like Lexical Matching).  

[1] Medelyan, O., Frank, E., & Witten, I. H. (2009). Human-competitive tagging using automatic keyphrase extraction. ACL and AFNLP, 6–7. https://doi.org/10.5555/3454287.3454810

[2] Frank, E., Paynter, G. W., Witten, I. H., Gutwin, C., & Nevill-Manning, C. G. (1999). Domain-Specific Keyphrase Extraction. Proceedings of the 16 Th International Joint Conference on Artifical Intelligence (IJCAI99), 668–673.


## Why embedding based matching

Existing subject indexing methods are roughly categorized into lexical matching algortihms and statistical learning algorithms. Lexical matching algorithms search for occurences of subjects from the controlled vocabulary over a given input text on the basis of their string representation. Statistical learning tries to learn patterns between input texts and gold standard annotations from large training corpora. 

Statistical learning can only predict subjects that have occured in the gold standard annotations used for training. It is uncapable of zero shot predictions. Lexical matching can find any subjects that are part of the vocabulary. Unfortunately, lexical matching often produces a large amount of false positives, as matching input texts and vocabulary solely on their string representation does not capture any semantic context. In particular, disambiguation of subjects with similar string representation is a problem.

The idea of embedding based matching is to enhance lexcial matching with the power of sentence transformer embeddings. These embeddings can capture the semantic context of the input text and allow a vector based matching that does not (solely) rely on the string representation. 

Benefits of Embedding Based Matching:

  * strong zero shot capabilities
  * handling of synonyms and context

Disadvantages:

  * creating embeddings for longer input texts with many chunks can be computationally expensive
  * no generelization capabilities: statisticl learning methods can learn the usage of a vocabulary
     from large amounts of training data and therefore learn associations between patterns in input
     texts and vocabulary items that are beyond lexical matching or embedding similarity.
     Lexical matching and embedding based matching will always stay close to the text.  

## Ranker model

The ranker model copies the idea taken from lexical matching Algorithms like MLLM or Maui, that subject candidates
can be ranked based on additional context information, e.g.

  * `first_occurence`, `last_occurence`, `spread`: position (chunk number) of the subject match in a text 
  * `occurences`: number of occurence in a text
  * `score`: sum of the similarity scores of all matches between a text chunk's embeddings and label embeddings 
  * `is_PrefLabelTRUE`: pref-Label or alt-Label tags in the SKOS Vocabulary 

These are numerical features that can be used to train a **binary** classifier. Given a
few hundred examples with gold standard labels, the ranker is trained to 
predict if a suggested candidate label is indeed a match, based on the
numerical features collected during the matching process.  In contrast to
the complex extreme multi label classification problem, this is a a much simpler
problem to train a classifier for, as the selection of features that the binary classifier 
is trained on, does not depend on the particular label. 

Our ranker model is implemented using the [xgboost](https://xgboost.readthedocs.io/en/latest/index.html) library.

The following plot shows a variable importance plot of the xgboost Ranker-Model:


## Embedding model

Our code uses [Jina AI Embeddings](https://huggingface.co/jinaai/jina-embeddings-v3). 
These implement a technique known as Matryoshka Embedding that allows you to
flexibly choose the dimension of your embedding vectors, to find your own 
cost-performance trade off. 

In this demo application we use assymetric embeddings finetuned for retrieval: 
Embeddings of task `retrieval.query` for embedding the vocab and embeddings of task
`retrieval.passage` for embedding the text chunks. 

## Vector storage

This project uses DuckDB (https://duckdb.org/) as storage for the vocabulary and the generated embeddings as well as one of its extensions (DuckDB's Vector Similarity Search Extension - https://duckdb.org/docs/extensions/vss.html) for indexing and querying the embeddings. 
Benefits of duckdb are: 

  * it is served as a one-file database: no independent database server needed
  * it implements vectorized HNSW-Search
  * it allows parallel querying from multiple threads

In other words: duckdb allows a parallized vectorized vector search enabling 
highly efficient subject retrieval even across large subject ontologies and
also with large text corpora and longer documents. 

This VSS-extension allows for some configurations regarding the HNSW index and the choice of distance metric (see documentaion for details). In this project, the 'cosine' distance and the corresponding 'array_cosine_distance' function are used. The metric and function must be explicitly specified when creating and using the index and must match in order to work. To save the created index, the configuration option for the database 'hnsw_enable_experimental_persistence=true' must be set. This is not recommended by duckdb at the moment, but should not be a problem for this project as no further changes are expected once the collection has been created. Relevant and useful blog posts on the VSS Extension extension can be found here 
- https://duckdb.org/2024/05/03/vector-similarity-search-vss.html
- https://duckdb.org/2024/10/23/whats-new-in-the-vss-extension.html

## Usage 

The main entry point for the package is the class `ebm_model` and its methods. 
