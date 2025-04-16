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


# Here we test that the login endpoint works
@app.post("/login")
async def login(token: Union[str, None] = Header(alias="Authorization",
                                                 default=None)):
    # TODO: Implement proper authentication
    #if token == None:
    #    raise HTTPException(status_code=401, detail="Credentials not provided")
    return {"id-token": "mock-token", "refresh-token": "mock-refresh-token"}


# Here we expose a way to post jobs,
# Must have a Access Token
# with entry_point tag
@app.post("/v1.0.0/jobs")
async def postJob(job: Job,
                  token: Union[str, None] = Header(alias="Authorization",
                                                   default=None)):
    # TODO: Implement job submission
    #if token == None:
    #    raise HTTPException(status_code=401, detail="Credentials not provided")
    return {"job_ids": ["mock-job-id"]}


# Retrieve the job, simulate having to wait by counting to 3
# until we return the job results
@app.get("/v1.0.0/job/{id}")
async def getJob(id: str):
    # TODO: Implement job retrieval
    return {
        "status": "Done",
        "samples": {
            "00": 50,
            "11": 50
        }
    }


def startServer(port):
    uvicorn.run(app, port=port, host='0.0.0.0', log_level="info")


if __name__ == '__main__':
    startServer(62448)