# Pinecone

To set up the environment for the first time

[how_to_run_first_time.md](how_to_run_first_time.md)

If virtual environment already exists follow: [how_to_run.md](how_to_run.md)

## Overview

This lab focuses on the following key points:
- Extending LLMs with custom functionality via function-calling, enabling them to form calls to external functions.
- Extracting structured data from natural language inputs, making real-world data usable for analysis.

## Lessons

### Lesson 1: Pinecone quickstart
- Creating a vector index, store and search through the vectors

### Lesson 2: Interacting with pinecone
- Pinecone creates an index for input vectors, and it allows querying the  nearest neighbors. A Pinecone index supports the following operations:
   - upsert: insert data formatted as (id, vector) tuples into the index, or replace existing (id, vector) tuples with new vector values. Optionally, you can attach metadata for each vector so you can use them in the query by specifying conditions. The upserted vector will look like (id, vector, metadata).
   - delete: delete vectors by id.
   - query: query the index and retrieve the top-k nearest neighbors based on dot-product, cosine-similarity, Euclidean distance, and more.
   - fetch: fetch vectors stored in the index by id.
   - describe_index_stats: get statistics about the index. 

### Lesson 3: Metadata filtering with pinecone
- Metadata filtering is a new feature in Pinecone that allows to apply filters on vector search based on metadata. Metadata can be added to the embeddings within Pinecone, and then filter for those criteria when sending the query.

### Lesson 4: Namespacing with pinecone
- Namespacing is a feature in a Pinecone service that allows to partition the data in an index. When you read from or write to a namespace in an index, you only access data in that particular namespace.

### Lesson 5: Building a Simple Classifier with Pinecone
- Building a simple nearest neighbor classifier
