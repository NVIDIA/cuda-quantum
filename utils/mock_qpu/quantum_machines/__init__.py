# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from fastapi import FastAPI, HTTPException, Header
from typing import Union
import uuid
from pydantic import BaseModel
import logging
import copy

# Define the REST Server App
app = FastAPI()


class Input(BaseModel):
    format: str
    data: str


# Jobs look like the following type
class Job(BaseModel):
    shots: int
    content: str
    executor: str
    qubit_mapping: str = None
    api_key: str = None
    source: str = "oq2"


createdJobs = {}

# Could how many times the client has requested the Job
countJobGetRequests = 0

# Save how many qubits were needed for each test (emulates real backend)
numQubitsRequired = 0

server_exec_response = {
    "id": "12345678-1234-1234-1234-0123456789ab",
    "samples": {
        "000": 19,
        "001": 2,
        "010": 27,
        "011": 4,
        "100": 11,
        "101": 3,
        "110": 30,
        "111": 4
    },
    "status": "Done"
}


@app.post("/v1/execute")
async def post_execute_job(job: Job,
                           token: Union[str,
                                        None] = Header(alias="Authorization",
                                                       default=None)):
    global createdJobs
    logging.info("In /v1/execute. code: {}", job)
    jobID = uuid.uuid4()
    response = copy.deepcopy(server_exec_response)
    response['id'] = jobID
    createdJobs[jobID] = response
    logging.info("In /v1/execute. response: {}", response)
    return response


# Retrieve the job, simulate having to wait by counting to 3
# until we return the job results
@app.get("/v1/results/{id}")
async def get_results(id: str):
    global countJobGetRequests, createdJobs
    if countJobGetRequests <= 3:
        countJobGetRequests += 1
        logging.info("In /v1/results/{}. countJobGetRequests: {}", id,
                     countJobGetRequests)
        return {"status": "InProgress"}
    countJobGetRequests = 0
    response = copy.deepcopy(server_exec_response)
    response['id'] = id
    logging.info("In /v1/results/{}. returning job results: {}", id, response)
    assert response
    return response
