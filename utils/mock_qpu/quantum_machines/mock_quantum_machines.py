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
    openqasm_code : str
    shots : int 

# Keep track of Job Ids to their Names
createdJobs = {}

# Could how many times the client has requested the Job
countJobGetRequests = 0

# Global holding the number of shots
shots = 100

# Here we expose a way to post jobs, 
@app.post("/execute")
async def postJob(job : Job):
    global createdJobs, shots
    
    code = job.openqasm_code
    shots = job.shots
    newId = str(uuid.uuid4())

    # Job "created", return the id
    return {"execution_id":newId}

# Retrieve the job, simulate having to wait by counting to 3 
# until we return the job results
@app.get("/results")  
async def getJob(request : Request):
    global countJobGetRequests, createdJobs, shots
    jobId = request.query_params._dict["execution_id"]
    print("request job id ", jobId)
    if countJobGetRequests < 3:
        countJobGetRequests += 1
        return {"status":"running"}

    countJobGetRequests = 0
   
    return {"counts":{"_00000":22,"_01000":22}}
   
def startServer(port):
    uvicorn.run(app, port=port, host='0.0.0.0', log_level="info")

if __name__ == '__main__':
    startServer(62234)