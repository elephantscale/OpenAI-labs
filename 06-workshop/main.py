from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from Langchain_Semantic_Search_Pinecone import get_answer

app = FastAPI(
    title="MosesAI API",
    description="This API allows to ask Talmud questions. \
                Click on the blue GET button, \
                Then click on 'Try it out', \
                Then ask your question. ",
    version="0.0.3",
)


class Item(BaseModel):
    question: str
    answer: str


@app.get("/items2/{question}", response_model=Item, summary="Read item", description="Read a specific item by its id.")
def read_item(question: str):
    answer = get_answer(question)
    return {"question": question, "answer": answer}


@app.get("/", summary="Read the root", description="Shalom. I am MosesAI, and right now, I know Talmud Illuminated")
def read_root():
    return {"Hello": "Shalom"}
