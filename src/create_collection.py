import weaviate
import weaviate.classes as wvc
import argparse

# Create argument parser
parser = argparse.ArgumentParser(description='Process command line arguments')

# Add arguments with default values
parser.add_argument('--task', type=str, default='title', help='Task name')
parser.add_argument('--evalset', type=str, default='test', help='Evaluation set')
parser.add_argument('--overwrite', type=lambda x: (str(x).lower() == 'true'), default=False, help='Overwrite collection if it already exists')
parser.add_argument('--collection_name_file', type=str, default='results/test/collection_name.txt', help='Output file for collection name')

# Parse arguments
args = parser.parse_args()

# create a weaviate collection for chunked documents
def create_collection(client, task_name, evalset, overwrite: bool = False, embeddings = 'baai_bge_m3'):
    collection_name = f'{task_name}_{evalset}_{embeddings}'
    if client.collections.exists(collection_name):
        print(f"Collection {collection_name} already exists")
        if overwrite:
            client.collections.delete(collection_name)
            print(f"Old Collection {collection_name} deleted")
        else:
            return collection_name

    client.collections.create(
        name=collection_name,
        properties=[
            wvc.config.Property(
                name="doc_id",
                description="DNB internal identifier",
                data_type=wvc.config.DataType.TEXT,
                tokenization=wvc.config.Tokenization.WORD,
                vectorize_property_name=False,
                skip_vectorization=True,
                index_searchable=False,
                index_filterable=True,
            ),
            wvc.config.Property(
                name="chunk_id",
                description="Chunk identifier",
                data_type=wvc.config.DataType.TEXT,
                tokenization=None,
                vectorize_property_name=False,
                skip_vectorization=True,
                index_searchable=False,
                index_filterable=True,
            ),
            wvc.config.Property(
                name="chunk_text",
                description="Chunk text",
                data_type=wvc.config.DataType.TEXT,
                vectorize_property_name=False,
                tokenization=wvc.config.Tokenization.WORD,
                index_searchable=True,
                index_filterable=False,
            ),
            wvc.config.Property(
                name="chunking_config",
                description="Metadta about chunking involved",
                data_type=wvc.config.DataType.TEXT,
                tokenization=wvc.config.Tokenization.WORD,
                vectorize_property_name=False,
                skip_vectorization=True,
                index_searchable=False,
                index_filterable=True
            )
        ],
        vectorizer_config=wvc.config.Configure.Vectorizer.none()
    )
    print(f"Collection {collection_name} created")

    return collection_name

client = weaviate.connect_to_local()
if not client.is_ready():
      sys.exit("Weaviate client is not ready. Exiting...")

collection_name = create_collection(client, args.task, args.evalset, overwrite=args.overwrite)

client.close()

# Write collection name to file
with open(args.collection_name_file, 'w') as file:
    file.write(collection_name)