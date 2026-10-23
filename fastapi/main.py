import logging
import sys
import time
import uuid
from pathlib import Path

from langchain_core.runnables import RunnableConfig

# Must run before any project import so Knowledge-Graph modules resolve
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import uvicorn
from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from common.config import Config
from common.logger_utils import get_logger, truncate

logger = get_logger("kg.api", "fastapi.log")

conf = Config()

logger.info("Loading LangGraph graph (LLM / Neo4j / FAISS)...")
_load_start = time.perf_counter()
from langgraph_nodes.langgraph_more_nodes import graph  # noqa: E402

logger.info("LangGraph graph loaded in %.2fs", time.perf_counter() - _load_start)

app = FastAPI(title=conf.FASTAPI_TITLE, description=conf.FASTAPI_DESCRIPTION)


class ProcessRequest(BaseModel):
    input: str = Field(..., description="User question")
    user_id: str = Field(
        ..., description="Unique user ID for isolating conversation history across users"
    )


class ProcessResponse(BaseModel):
    input: str
    output: str


@app.middleware("http")
async def log_requests(request: Request, call_next):
    # Reuse the client's request ID if it sent one so frontend and API logs can be matched up
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:8]
    request.state.request_id = request_id
    client = request.client.host if request.client else "-"
    logger.info("[%s] --> %s %s from %s", request_id, request.method, request.url.path, client)
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("[%s] Unhandled error on %s %s", request_id, request.method, request.url.path)
        raise
    response.headers["X-Request-ID"] = request_id
    logger.info(
        "[%s] <-- %s %s %d (%.2fs)",
        request_id, request.method, request.url.path, response.status_code, time.perf_counter() - start,
    )
    return response


@app.exception_handler(RequestValidationError)
async def log_validation_error(request: Request, exc: RequestValidationError):
    logger.warning(
        "[%s] Request validation failed: %s", getattr(request.state, "request_id", "-"), exc.errors()
    )
    return await request_validation_exception_handler(request, exc)


@app.get("/")
async def health_check():
    return {"status": "ok", "message": conf.FASTAPI_HEALTH_MESSAGE}


KEY_STATE_FIELDS = tuple(conf.get_json("FASTAPI_KEY_STATE_FIELDS", []) or [])


def run_graph(config: RunnableConfig, question: str) -> dict:
    """Run the graph node by node so every step and its state update gets logged."""
    configurable = config.get("configurable") or {}
    request_id = configurable.get("request_id", "-")
    state = {"input": question}
    # stream_mode="updates" yields {node_name: update} after each node; AgentState has no
    # reducers, so merging the updates in order reproduces graph.invoke's final state.
    # config.thread_id is required by InMemorySaver so history persists across turns.
    for step, update in enumerate(graph.stream(state, stream_mode="updates", config=config), start=1):
        for node_name, node_update in update.items():
            node_update = node_update or {}
            # Nodes may return the whole state, so only log the fields this step changed
            changed = {k: v for k, v in node_update.items() if k not in state or state[k] != v}
            logger.info("[%s] step %d: node '%s' done, changed=%s", request_id, step, node_name, list(changed))
            for key, value in changed.items():
                # Routing decisions are useful at INFO; everything else (entities, results) at DEBUG
                level = logging.INFO if key in KEY_STATE_FIELDS else logging.DEBUG
                logger.log(level, "[%s]     %s = %s", request_id, key, truncate(value))
            state.update(node_update)
    return state


@app.post("/process", response_model=ProcessResponse)
async def process(request: ProcessRequest, http_request: Request):
    request_id = http_request.state.request_id
    logger.info("[%s] /process user_id=%s input=%r", request_id, request.user_id, request.input)
    # thread_id scopes checkpointer history per user; request_id is only for log correlation
    config: RunnableConfig = {
        "configurable": {
            "thread_id": request.user_id,
            "request_id": request_id,
        }
    }
    start = time.perf_counter()
    try:
        # The graph is blocking (LLM / Neo4j / FAISS calls), keep it off the event loop
        result = await run_in_threadpool(run_graph, config, request.input)
        output = result.get("output", "")
        if not output:
            logger.warning("[%s] Graph finished with an empty output", request_id)
        logger.info(
            "[%s] /process done in %.2fs, output=%s",
            request_id, time.perf_counter() - start, truncate(output, 200),
        )
        return ProcessResponse(input=request.input, output=output)
    except Exception:
        logger.exception(
            "[%s] /process failed after %.2fs for input=%r",
            request_id, time.perf_counter() - start, request.input,
        )
        return JSONResponse(
            status_code=500, content={"input": "", "output": conf.FASTAPI_ERROR_OUTPUT}
        )


if __name__ == "__main__":
    logger.info("Starting FastAPI on %s:%d", conf.FASTAPI_HOST, conf.FASTAPI_PORT)
    uvicorn.run(app, host=conf.FASTAPI_HOST, port=conf.FASTAPI_PORT)
