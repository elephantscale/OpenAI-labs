# Main for MosesAI fastapi

from fastapi import FastAPI
from starlette.responses import RedirectResponse
from pydantic import BaseModel
from Talmud_read_write import get_answer
import logging

VERSION = "0.2.6"
# Configure logging
logging.basicConfig(filename='MosesAI.log', level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="MosesAI API",
    description="This API allows to ask Talmud questions. \
                Click on the blue GET button, \
                Then click on 'Try it out', \
                Then ask your question. ",
    version=VERSION
)


class Item(BaseModel):
    question: str
    answer: str


@app.get("/{question}", response_model=Item, summary="Ask Moses AI a question",
         description="Click on 'Try it out', then ask away!")
def read_item(question: str):
    answer = get_answer(question)
    logger.info("Q: " + question)
    logger.info("A: " + answer)
    return {"question": question, "answer": answer}


@app.get("/{question_with_options}/{num_sources}")
def read_item(question_with_options: str, num_sources: int):
    answer = get_answer(question_with_options, num_sources)
    logger.info("Q: " + question_with_options)
    logger.info("A: " + answer)
    return {"question": question_with_options, "answer": answer}


@app.get("/")
def read_root():
    logger.info("Shalom!!! Redirecting to /doc")
    return RedirectResponse(url='/docs')
