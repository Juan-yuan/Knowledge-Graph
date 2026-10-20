import sys
import traceback
from pathlib import Path

# Must run before any project import so Knowledge-Graph modules resolve
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import uvicorn
from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from common.config import Config
from langgraph_nodes.langgraph_more_nodes import graph

conf = Config()

app = FastAPI(title=conf.FASTAPI_TITLE, description=conf.FASTAPI_DESCRIPTION)


class ProcessRequest(BaseModel):
    input: str = Field(..., description="User question")
    user_id: str = Field(
        ..., description="Unique user ID for isolating conversation history across users"
    )


class ProcessResponse(BaseModel):
    input: str
    output: str


@app.get("/")
async def health_check():
    return {"status": "ok", "message": conf.FASTAPI_HEALTH_MESSAGE}


@app.post("/process", response_model=ProcessResponse)
async def process(request: ProcessRequest):
    try:
        # graph.invoke is blocking (LLM / Neo4j / FAISS calls), keep it off the event loop
        result = await run_in_threadpool(graph.invoke, {"input": request.input})
        return ProcessResponse(input=request.input, output=result.get("output", ""))
    except Exception:
        traceback.print_exc()
        return JSONResponse(
            status_code=500, content={"input": "", "output": conf.FASTAPI_ERROR_OUTPUT}
        )


if __name__ == "__main__":
    uvicorn.run(app, host=conf.FASTAPI_HOST, port=conf.FASTAPI_PORT)
