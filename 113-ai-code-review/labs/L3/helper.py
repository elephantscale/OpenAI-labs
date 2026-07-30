import html
import os
import sqlite3
import time
from pathlib import Path

import pandas as pd
from IPython.display import HTML
from cerebras.cloud.sdk import APIStatusError, Cerebras, RateLimitError
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import ValidationError



load_dotenv()


# Shared environment and client helpers
def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Set {name} in .env before running this notebook.")
    return value


def cerebras_client(api_key: str | None = None) -> Cerebras:
    return Cerebras(api_key=api_key or require_env("CEREBRAS_API_KEY"))


def openai_client(api_key: str | None = None) -> OpenAI:
    return OpenAI(api_key=api_key or require_env("OPENAI_API_KEY"))

# Shared response formatting
def usage_metrics(response, elapsed: float, fallback_tokens: int) -> dict:
    usage = getattr(response, "usage", None)
    completion_tokens = getattr(usage, "completion_tokens", None) or fallback_tokens

    time_info = getattr(response, "time_info", None)
    completion_time = getattr(time_info, "completion_time", None) if time_info else None
    total_time = getattr(time_info, "total_time", None) if time_info else None

    completion_time = completion_time or elapsed
    total_time = total_time or elapsed

    return {
        "completion_tokens": completion_tokens,
        "completion_time": round(completion_time, 3),
        "duration": round(total_time, 3),
        "latency_ms": round(total_time * 1000, 1),
        "tps": round(completion_tokens / completion_time, 1) if completion_time > 0 else 0.0,
    }


def _cerebras_request_with_retry(client: Cerebras, request: dict, *, max_attempts: int = 4):
    retrying_client = client.with_options(max_retries=0)
    retry_wait_s = 0.0

    for attempt in range(1, max_attempts + 1):
        try:
            response = retrying_client.chat.completions.create(**request)
            return response, retry_wait_s
        except (RateLimitError, APIStatusError) as err:
            status_code = getattr(err, "status_code", None)
            if status_code != 429 or attempt == max_attempts:
                raise
            wait_s = min(8.0, float(2 ** (attempt - 1)))
            time.sleep(wait_s)
            retry_wait_s += wait_s

# Lesson 2, Lesson 4, and Lesson 5 helpers
def call_cerebras(
    client: Cerebras | None = None,
    model: str | None = None,
    prompt: str | None = None,
    *,
    user_prompt: str | None = None,
    system_prompt: str | None = None,
    api_key: str | None = None,
    schema_model=None,
    max_tokens: int = 220,
    max_completion_tokens: int | None = None,
    temperature: float = 0.3,
    stream: bool = False,
):
    resolved_model = model
    if resolved_model is None:
        raise ValueError("model is required.")

    resolved_prompt = user_prompt if user_prompt is not None else prompt
    if resolved_prompt is None:
        raise ValueError("prompt or user_prompt is required.")

    client = client or cerebras_client(api_key)
    messages = [{"role": "user", "content": resolved_prompt}]
    if system_prompt:
        messages.insert(0, {"role": "system", "content": system_prompt})

    token_budget = max_completion_tokens if max_completion_tokens is not None else max_tokens
    request = {
        "model": resolved_model,
        "messages": messages,
        "max_completion_tokens": token_budget,
        "temperature": temperature,
        "stream": stream,
    }

    if schema_model is not None:
        request["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": schema_model.__name__.lower(),
                "strict": True,
                "schema": schema_model.model_json_schema(),
            },
        }

    if schema_model is None:
        start = time.perf_counter()
        response, retry_wait_s = _cerebras_request_with_retry(client, request)
        elapsed = max(0.0, time.perf_counter() - start - retry_wait_s)
        content = response.choices[0].message.content or ""
        metrics = usage_metrics(response, elapsed, max(1, len(content) // 4))
        return {
            "provider": "Cerebras",
            "response": content,
            **metrics,
        }

    budgets = [token_budget]
    for candidate in (
        max(token_budget + 200, int(token_budget * 1.5)),
        max(token_budget + 400, token_budget * 2),
    ):
        if candidate not in budgets:
            budgets.append(candidate)

    last_error = None

    for budget in budgets:
        request["max_completion_tokens"] = budget
        start = time.perf_counter()
        response, retry_wait_s = _cerebras_request_with_retry(client, request)
        elapsed = max(0.0, time.perf_counter() - start - retry_wait_s)
        content = response.choices[0].message.content

        if content is None:
            last_error = ValueError("Model returned no structured content.")
            continue

        metrics = usage_metrics(response, elapsed, max(1, len(content) // 4))

        try:
            return schema_model.model_validate_json(content), metrics
        except ValidationError as err:
            last_error = err

    assert last_error is not None
    raise last_error




from pydantic import BaseModel, ConfigDict

from typing import Literal
class StructuredModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

class CompanyDatabaseContext(StructuredModel):
    found_in_db: bool
    target_company: Literal["chipmaker", "automotive_company", "media_company", "other"]
    previous_decision: Literal["buy", "sell", "watch", "none"]
    last_reviewed_at: str | None
    facts: list[str]

# Lesson 6 helpers
def load_database(db_path: Path | str, target_company: str, limit: int = 5):
    connection = sqlite3.connect(f"file:{Path(db_path)}?mode=ro", uri=True)
    query = """
        SELECT company_alias, activity_timestamp, activity_type, fact, previous_decision, analyst_note
        FROM company_memory
        WHERE company_alias = ?
        ORDER BY rowid DESC
        LIMIT ?
    """
    frame = pd.read_sql_query(query, connection, params=(target_company, limit))
    connection.close()
    context = CompanyDatabaseContext(
        found_in_db=True,
        target_company=target_company,
        previous_decision=frame.iloc[0]["previous_decision"],
        last_reviewed_at=frame.iloc[0]["activity_timestamp"],
        facts=frame["fact"].tolist(),
    )

    return context, frame


__all__ = [
    "call_cerebras",
    "call_openai",
    "load_database",
    "render_investment_decision_card",
    "usage_metrics",
]


