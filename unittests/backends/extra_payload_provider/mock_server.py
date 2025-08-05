# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
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
    if "unicorn" not in payload:
        print("No unicorn in payload")
        raise HTTPException(status_code=401,
                            detail="Cannot find the extra payload")

    newId = str(uuid.uuid4())
    # Job "created", return the id
    return ({"job_token": newId}, 201)


# Retrieve the job, simulate having to wait by counting to 3
# until we return the job results
@app.get("/job/{jobId}")
async def getJob(jobId: str):
    res = ({"status": "done"}, 201)
    return res


def startServer(port):
    print("Server Started")
    uvicorn.run(app, port=port, host='0.0.0.0', log_level="debug")


if __name__ == '__main__':
    print("Server Starting")
    startServer(56686)
