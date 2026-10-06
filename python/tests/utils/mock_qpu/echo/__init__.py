# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import uuid
from typing import Any

from fastapi import FastAPI, HTTPException, Request

app = FastAPI()

createdJobs: dict[str, Any] = {}


async def read_payload(request: Request) -> Any:
    body = await request.body()
    if not body:
        return None
    try:
        return await request.json()
    except Exception:
        return body.decode()


def extract_value(payload: Any) -> Any:
    if isinstance(payload, dict) and "value" in payload:
        return payload["value"]
    return payload


@app.post("/{path:path}")
async def post_job(path: str, request: Request):
    value = extract_value(await read_payload(request))
    job_id = str(uuid.uuid4())
    createdJobs[job_id] = value
    return {"id": job_id, "status": "done", "value": value}


@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    if job_id not in createdJobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"id": job_id, "status": "done", "value": createdJobs[job_id]}


def startServer(port):
    import uvicorn
    uvicorn.run(app, port=port, host="0.0.0.0", log_level="info")
