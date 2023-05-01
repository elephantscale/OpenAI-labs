# Embeddings

https://platform.openai.com/docs/guides/embeddings

* In this lab we will practice how to use embeddings to do semantic search

An embedding is a vector (list) of floating point numbers. The distance between two vectors measures their relatedness. Small distances suggest high relatedness and large distances suggest low relatedness.

### Lab Goals:

* Run semantic searches in a PDF text content

### Requirements:

1. Python installed on your machine
2. Valid OpenAI-API key


### Step 1) Prepare environment

* Install OpenAI package

``` bash
pip install --upgrade openai
```

Note: Replace <OPENAI_API_KEY> with the API key provided

* Install pandas package

``` bash
pip install --upgrade pandas
```

* Install gradio package to do a simple UI to ask questions

``` bash
pip install --upgrade gradio
```

* Install langchain package to extract text from the Pdf file

``` bash
pip install langchain pypdf
```

* Install OpenAi embeddings dependencies

``` bash
pip install sklearn
pip install plotly
pip install scipy
```

### Step 2) Search over a PDF file


* Create a new python script 

``` python
import gradio as gr
import openai
import pandas as pd
import plotly.express as px

from openai.embeddings_utils import get_embedding
from openai.embeddings_utils import cosine_similarity
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter

openai.api_key = "<OPENAI_API_KEY>"
```

Note: Replace <OPENAI_API_KEY> with the API key provided

* Split the pdf content in chucks (In this case we are splitting by paragraph.)

``` python
oader = PyPDFLoader("<PDF_PATH>")
pages = loader.load_and_split()

split = CharacterTextSplitter(chunk_size=300, separator = '.\n')
texts = split.split_documents(pages) 

texts = [str(i.page_content) for i in texts] 
paragraphs = pd.DataFrame(texts, columns=["text"])

paragraphs['Embedding'] = paragraphs["text"].apply(lambda x: get_embedding(x, engine='text-embedding-ada-002'))
paragraphs.to_csv('embeddings.csv')
```

Note: Replace <PDF_PATH> with the path of the pdf you want to perform semantic search.

* Get the embeddings per each paragraph and stores it in a CSV.
  * If you want to run multiple search over the same pdf file, it is recommended store the embeddings data into a csv and re use it every time you want to perform a search, instead of get the embeddings from OpenAI every time.

``` python
def embed_text(path="text.csv"):
knowledge_df = pd.read_csv(path)
knowledge_df['Embedding'] = knowledge_df['text'].apply(lambda x: get_embedding(x, engine='text-embedding-ada-002'))
knowledge_df.to_csv('mtg-embeddings.csv')
return knowledge_df

def search(search, data, n_results=5):
search_embed = get_embedding(search, engine="text-embedding-ada-002")
data["Similarity"] = data['Embedding'].apply(lambda x: cosine_similarity(x, search_embed))
data = data.sort_values("Similarity", ascending=False)
return data.iloc[:n_results][["text", "Similarity", "Embedding"]]

```

* Using the gradio package, we will build a simple UI that will let put the search and display the results.

``` python
text_emb = paragraphs
with gr.Blocks() as demo:
searchText = gr.Textbox(label="Search")
output = gr.DataFrame(headers=['text'])
greet_btn = gr.Button("Send!")
greet_btn.click(fn=search, inputs=[searchText, gr.DataFrame(text_emb)], outputs=output)

demo.launch()

```

### Step 3) Test your embeddings

* Run the python script

  - The site will be available using http://127.0.0.1:7860

![img_embeddings.png](../images/img_embeddings.png)

## Congratulations!
