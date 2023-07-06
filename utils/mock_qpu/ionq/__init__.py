# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from fastapi import FastAPI, Request, HTTPException, Header
from typing import Optional, Union
import uvicorn, uuid, json, base64
from pydantic import BaseModel
from llvmlite import binding as llvm

# Define the REST Server App
app = FastAPI()


# Jobs look like the following type
class Job(BaseModel):
    name: str
    program: str
    count: int


# Keep track of Job Ids to their Names
createdJobs = {}

# Could how many times the client has requested the Job
countJobGetRequests = 0

# Global holding the number of shots
shots = 100


# Here we test that the login endpoint works
@app.post("/login")
async def login(token: Union[str, None] = Header(alias="Authorization",
                                                 default=None)):
    if 'token' == None:
        raise HTTPException(status_code(401), detail="Credentials not provided")
    return {"id-token": "hello", "refresh-token": "refreshToken"}


# Here we expose a way to post jobs,
# Must have a Access Token, Job Program must be Adaptive Profile
# with EntryPoint tag
@app.post("/job")
async def postJob(job: Job,
                  token: Union[str, None] = Header(alias="Authorization",
                                                   default=None)):
    global createdJobs, shots

    if 'token' == None:
        raise HTTPException(status_code(401), detail="Credentials not provided")

    print('Posting job with name = ', job.name)
    name = job.name
    newId = str(uuid.uuid4())
    createdJobs[newId] = name
    shots = job.count
    program = job.program
    decoded = base64.b64decode(program)
    m = llvm.module.parse_bitcode(decoded)
    mstr = str(m)
    assert ('EntryPoint' in mstr)

    if name == "XX":
        assert ("qis__h__body" in mstr)
    elif name == "YY":
        assert ("qis__ry__body" in mstr and "qis__mz__body" in mstr)
    elif name == "ZI":
        assert ("qis__mz__body" in mstr)
    elif name == "IZ":
        assert ("qis__mz__body" in mstr)

    # Job "created", return the id
    return {"job": newId}


# Retrieve the job, simulate having to wait by counting to 3
# until we return the job results
@app.get("/job/{jobId}")
async def getJob(jobId: str):
    global countJobGetRequests, createdJobs, shots

    if countJobGetRequests < 3:
        countJobGetRequests += 1
        return {"status": "running"}

    countJobGetRequests = 0
    name = createdJobs[jobId]
    retData = []
    if name == "XX":
        retData = ['11'] * 3887 + ['10'] * 1104 + ['01'] * 1095 + ['00'] * 3914
    elif name == "YY":
        retData = ['11'] * 3861 + ['10'] * 1104 + ['01'] * 1095 + ['00'] * 3914
    elif name == "ZI":
        retData = ['1'] * 9088 + ['0'] * 912
    elif name == "IZ":
        retData = ['1'] * 880 + ['0'] * 9120
    else:
        retData = ['00'] * int(shots / 2) + ['11'] * int(shots / 2)
    res = {"status": "completed", "results": {"mz0": retData}}
    return res


def startServer(port):
    uvicorn.run(app, port=port, host='0.0.0.0', log_level="info")


if __name__ == '__main__':
    startServer(62455)
