# Function-Calling and Data Extraction with NexusRavenV2-13B

## Setup Local Environment for Lab 29 - Pinecone

1. **Navigate to the Project Directory**
   - Go inside the `29-pinecone` folder.

2. **Create a Virtual Environment**
   - Run the following command to create a virtual environment:
     ```sh
     python3 -m venv 29
     ```

3. **Activate the Virtual Environment**
   - Activate the new virtual environment using:
     ```sh
     source 29/bin/activate
     ```

4. **Install Jupyter Notebook**
   - Install Jupyter Notebook in the same virtual environment:
     ```sh
     pip3 install jupyter
     ```

5. **Install IPython Kernel**
   - Install `ipykernel` to attach the Jupyter environment to the same kernel:
     ```sh
     pip3 install ipykernel
     ```

6. **Add Environment to Jupyter Kernel**
   - Add the current environment to the Jupyter kernel:
     ```sh
     python3 -m ipykernel install --user --name=29 --display-name "Python (lab 29)"
     ```

7. **Install Lab Requirements**
   - Install the lab requirements specified in the `requirements.txt` file:
     ```sh
     pip3 install -r requirements.txt
     ```

8. **Create .env File for API Keys**
   - Create a `.env` file for the keys and add the following values:
     ```sh
     PINECONE_API_KEY=
     PINECONE_REGION=
     PINECONE_CLOUD=
     ```

9. **Start Jupyter Notebook**
   - Start Jupyter Notebook with the command:
     ```sh
     jupyter notebook
     ```

10. **Select the Correct Kernel**
    - At the top right corner of the Jupyter Notebook, click on the kernel and choose "Python (lab 28)".

11. **Verify the Environment**
    - To verify that you are in the correct environment, create a new cell and run the following command. If the output shows your current environment, you are all set:
      ```python
      import sys
      print(sys.executable)
      ```

12. **Run the Labs**
    - You are now ready to run any lab in Jupyter Notebook.

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
