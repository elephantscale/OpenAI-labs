# Main for MosesAI fastapi

from fastapi import FastAPI
from starlette.responses import RedirectResponse
from pydantic import BaseModel
from TI_read_write import get_answer

app = FastAPI(
    title="MosesAI API",
    description="This API allows to ask Talmud questions. \
                Click on the blue GET button, \
                Then click on 'Try it out', \
                Then ask your question. ",
    version="0.1.1",
)


class Item(BaseModel):
    question: str
    answer: str


@app.get("/{question}", response_model=Item, summary="Ask Moses AI a question",
         description="Click on 'Try it out', then ask away!")
def read_item(question: str):
    answer = get_answer(question)
    with open('OpenAI.log', 'a') as f:
        f.write("Q: " + question + "\n")
        f.write("A: " + answer + "\n")
    return {"question": question, "answer": answer}


@app.get("/{question_with_options}/{num_sources}")
def read_item(question_with_options: str, num_sources: int):
    answer = get_answer(question_with_options, num_sources)
    with open('OpenAI.log', 'a') as f:
        # Write the string to the file
        f.write("O: " + str(num_sources) + "\n")
        f.write("Q: " + question_with_options + "\n")
        f.write("A: " + answer + "\n")
    return {"question": question_with_options, "answer": answer}


@app.get("/")
def read_root():
    return RedirectResponse(url='/docs')
