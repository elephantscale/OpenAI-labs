import os
import openai
from dotenv import load_dotenv, find_dotenv
import pinecone
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings

_ = load_dotenv(find_dotenv())  # read local .env file
openai.api_key = os.getenv('OPENAI_API_KEY')
MODEL = "gpt-4"
embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)

query_result = embeddings.embed_query("Hello world")
print(len(query_result))

# initialize pinecone
pinecone.init(
    api_key=os.getenv('PINECONE_API_KEY'),
    environment="us-west4-gcp-free"  # next to api key in console
)

index_name = "langchain-demo"

# index = Pinecone.from_documents(docs, embeddings, index_name=index_name)

# if you already have an index, you can load it like this
index = Pinecone.from_existing_index(index_name, embeddings)


def get_similiar_docs(query, k=5, score=False):
    if score:
        similar_docs = index.similarity_search_with_score(query, k=k)
    else:
        similar_docs = index.similarity_search(query, k=k)
    return similar_docs


query = "When do you say Shema?"
similar_docs = get_similiar_docs(query)
len(similar_docs)

from langchain.llms import OpenAI

llm = OpenAI(model_name=MODEL)

from langchain.chains.question_answering import load_qa_chain

chain = load_qa_chain(llm, chain_type="stuff")


def get_answer(query):
    similar_docs = get_similiar_docs(query)
    # print(similar_docs)
    answer = chain.run(input_documents=similar_docs, question=query)
    return answer


query = "When to say Shema?"
answer = get_answer(query)
print(answer)
