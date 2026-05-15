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


# Keep track of Job Ids to their Names
createdJobs = {}

# Could how many times the client has requested the Job
countJobGetRequests = 0

# Save how many qubits were needed for each test (emulates real backend)
numQubitsRequired = 0


# Here we test that the login endpoint works
@app.post("/login")
async def login(token: Union[str, None] = Header(alias="Authorization",
                                                 default=None)):
    if token == None:
        raise HTTPException(status_code(401), detail="Credentials not provided")
    return {"id-token": "hello", "refresh-token": "refreshToken"}


# Here we expose a way to post jobs,
# Must have a Access Token
# with entry_point tag
@app.post("/v0.2.0/jobs")
async def postJob(job: Job,
                  token: Union[str, None] = Header(alias="Authorization",
                                                   default=None)):
    global createdJobs, shots, numQubitsRequired

    if token == None:
        raise HTTPException(status_code(401), detail="Credentials not provided")

    print('Posting job with shots = ', job.shots)
    newId = str(uuid.uuid4())
    # Job "created", return the id
    return {"job_ids": [newId]}


# Retrieve the job, simulate having to wait by counting to 3
# until we return the job results
@app.get("/v0.2.0/job/{id}")
async def getJob(id: str):
    print("Getting Job")
    global countJobGetRequests, createdJobs, numQubitsRequired

    # Simulate asynchronous execution
    if countJobGetRequests < 3:
        countJobGetRequests += 1
        return {"status": "running"}

    countJobGetRequests = 0
    res = {
        'status': 'Done',
        "samples": {
            "11": 49,
            "00": 51
        },
    }
    return res
