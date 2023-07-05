from starlette.responses import RedirectResponse
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from Talmud_read_write import get_answer
from typing import List, Optional
import datetime
import logging

VERSION = "0.3.1"
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


class QuestionWithHistory(BaseModel):
    question: str
    history: Optional[List[str]] = None


class Answer(BaseModel):
    answer: str


def get_answer_with_history(question: str, history: Optional[List[str]]) -> str:
    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if history is None:
        # Handle case where history is not provided
        logger.info(f"{current_time} - Question: {question} with no history")
        answer = get_answer(question)
        return answer
    else:
        # Implement your logic here to process the question and history to produce an answer.
        logger.info(f"{current_time} - Question: {question}")
        logger.info(f"{current_time} - History: {history}")

    return "This is a placeholder answer."


@app.post("/ask", response_model=Answer)
def ask_question(question_with_history: QuestionWithHistory):
    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logger.info(f"{current_time} - /ask")
    answer = get_answer_with_history(question_with_history.question, question_with_history.history)
    return {"answer": answer}


origins = [
    "http://localhost:8000",
    "http://localhost:3000"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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

