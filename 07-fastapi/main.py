from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

app = FastAPI(
    title="MosesAI API",
    description="This API allows to ask Talmud questions",
    version="0.0.2",
)


class Item(BaseModel):
    question: str
    answer: str

@app.get("/", summary="Read the root", description="Shalom. I am MosesAI, and right now, I know Talmud")
def read_root():
    return {"Hello": "Shalom"}

@app.get("/items2/{question}", response_model=Item, summary="Read item", description="Read a specific item by its id.")
def read_item(question: str, q: str = None):
    answer = "Your question was: " + question + ", and the answer is bla-bla-bla"
    return {"question": question, "answer": answer}
