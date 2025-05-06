# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from fastapi import FastAPI, HTTPException, Header
from typing import Union
import uvicorn, uuid
from pydantic import BaseModel

from python import cudaq

# Define the REST Server App
app = FastAPI()


class Input(BaseModel):
    format: str
    data: str


# Jobs look like the following type
class Job(BaseModel):
    qasm_strs: str
    shots: int
    target: str
    method: str = None


# TODO: Implement proper job tracking
createdJobs = {}

# Could how many times the client has requested the Job
countJobGetRequests = 0

# Save how many qubits were needed for each test (emulates real backend)
numQubitsRequired = 0

server_exec_response = {
    "id":"12345678-1234-1234-1234-0123456789ab",
    "samples":{"000":19,"001":2,"010":27,"011":4,"100":11,"101":3,"110":30,"111":4},
    "status":"Done"}

# Here we test that the login endpoint works
@app.post("/v1/auth")
async def login(token: Union[str, None] = Header(alias="Authorization",                                                 default=None)):
    # TODO: Implement proper authentication
    #if token == None:
    #    raise HTTPException(status_code=401, detail="Credentials not provided")
    return {"id-token": "mock-token", "refresh-token": "mock-refresh-token"}


@app.post("/v1/execute")
async def post_execute_job(job: Job,
                           token: Union[str, None] = Header(alias="Authorization",
                                                          default=None)):
    createdJobs[server_exec_response["id"]] = server_exec_response
    return server_exec_response

# Retrieve the job, simulate having to wait by counting to 3
# until we return the job results
@app.get("/v1/results/{id}")
async def get_job(id: str):
    # TODO: Implement job retrieval
    return createdJobs.get(id)


def start_server(port):
    uvicorn.run(app, port=port, host='0.0.0.0', log_level="info")


if __name__ == '__main__':
    start_server(62448)