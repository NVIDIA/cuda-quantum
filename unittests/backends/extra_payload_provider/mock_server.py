# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq
from fastapi import FastAPI, HTTPException, Header, Request
from typing import Union
import uvicorn, uuid, base64, ctypes
from pydantic import BaseModel

# Define the REST Server App
app = FastAPI()


@app.post("/job")
async def postJob(request: Request):
    payload = await request.json()
    print("Received payload:", payload)
    if "foo" not in payload:
        raise HTTPException(status_code=400,
                            detail="Payload must contain 'foo' key")
        if "bar" not in payload["foo"]:
            raise HTTPException(status_code=400,
                                detail="Payload must contain 'foo/bar' key")
        if payload["foo"]["bar"] != "test":
            raise HTTPException(status_code=400,
                                detail="Invalid value for 'foo/bar' key")
    newId = str(uuid.uuid4())
    # Job "created", return the id
    return ({"job_token": newId}, 201)


@app.get("/job/{jobId}")
async def getJob(jobId: str):
    res = ({"status": "done"}, 201)
    return res


def startServer(port):
    print("Server Started")
    uvicorn.run(app, port=port, host='0.0.0.0', log_level="debug")


if __name__ == '__main__':
    print("Server Starting")
    startServer(62450)
