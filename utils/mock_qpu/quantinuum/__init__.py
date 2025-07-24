# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq
from fastapi import FastAPI, HTTPException, Header, Query
from fastapi.responses import JSONResponse
from typing import Union
import uvicorn, uuid, base64, ctypes
from pydantic import BaseModel
from llvmlite import binding as llvm

# Define the REST Server App
app = FastAPI()


# Jobs look like the following type
class Job(BaseModel):
    name: str
    program: str
    count: int


# Keep track of the QIR module IDs to their names
createdQIRModules = {}

# Keep track of Job Ids to their Names
createdJobs = {}

# Could how many times the client has requested the Job
countJobGetRequests = 0

llvm.initialize()
llvm.initialize_native_target()
llvm.initialize_native_asmprinter()
target = llvm.Target.from_default_triple()
targetMachine = target.create_target_machine()
backing_mod = llvm.parse_assembly("")
engine = llvm.create_mcjit_compiler(backing_mod, targetMachine)


def getKernelFunction(module):
    for f in module.functions:
        if not f.is_declaration:
            return f
    return None


def getNumRequiredQubits(function):
    for a in function.attributes:
        if "requiredQubits" in str(a):
            return int(
                str(a).split("requiredQubits\"=")[-1].split(" ")[0].replace(
                    "\"", "").replace("'", ""))


# Here we test that the login endpoint works
@app.post("/login")
async def login(token: Union[str, None] = Header(alias="Authorization",
                                                 default=None)):
    if 'token' == None:
        raise HTTPException(status_code(401), detail="Credentials not provided")
    return {"id-token": "hello", "refresh-token": "refreshToken"}


# Here we expose a way to post jobs,
# Must have a Access Token, Job Program must be Adaptive Profile
# with entry_point tag
@app.post("/job")
async def postJob(job: Job,
                  token: Union[str, None] = Header(alias="Authorization",
                                                   default=None)):
    global createdJobs, shots

    if 'token' == None:
        raise HTTPException(status_code(401), detail="Credentials not provided")

    print('Posting job with name = ', job.name, job.count)
    name = job.name
    newId = str(uuid.uuid4())
    program = job.program
    decoded = base64.b64decode(program)
    m = llvm.module.parse_bitcode(decoded)
    mstr = str(m)
    assert ('entry_point' in mstr)

    # Get the function, number of qubits, and kernel name
    function = getKernelFunction(m)
    if function == None:
        raise Exception("Could not find kernel function")
    numQubitsRequired = getNumRequiredQubits(function)
    kernelFunctionName = function.name

    print("Kernel name = ", kernelFunctionName)
    print("Requires {} qubits".format(numQubitsRequired))

    # JIT Compile and get Function Pointer
    engine.add_module(m)
    engine.finalize_object()
    engine.run_static_constructors()
    funcPtr = engine.get_function_address(kernelFunctionName)
    kernel = ctypes.CFUNCTYPE(None)(funcPtr)

    # Invoke the Kernel
    cudaq.testing.toggleDynamicQubitManagement()
    qubits, context = cudaq.testing.initialize(numQubitsRequired, job.count)
    kernel()
    results = cudaq.testing.finalize(qubits, context)
    results.dump()
    createdJobs[newId] = (name, results)

    engine.remove_module(m)

    # Job "created", return the id
    return {"job": newId}


# Retrieve the job, simulate having to wait by counting to 3
# until we return the job results
@app.get("/job/{jobId}")
async def getJob(jobId: str):
    global countJobGetRequests, createdJobs, shots

    # Simulate asynchronous execution
    if countJobGetRequests < 3:
        countJobGetRequests += 1
        return {"status": "running"}

    countJobGetRequests = 0
    name, counts = createdJobs[jobId]
    retData = []
    for bits, count in counts.items():
        retData += [bits] * count

    # The simulators don't implement result recording features yet, so we have
    # to mark these results specially (MOCK_SERVER_RESULTS) in order to allow
    # downstream code to recognize that this isn't from a true Quantinuum QPU.
    res = {"status": "completed", "results": {"MOCK_SERVER_RESULTS": retData}}
    return res


## Nexus APIs
# Support for token refresh endpoint
@app.post("/auth/tokens/refresh")
async def refresh_tokens():
    response = {"message": "Token refreshed"}
    # Set cookie headers in response
    headers = {"Set-Cookie": "myqos_id=mock_api_key; Path=/;"}
    return JSONResponse(content=response, headers=headers)


# Project endpoints to support lookups
@app.get("/api/projects/v1beta2/{project_id}")
async def get_project_by_id(project_id: str):
    # Return mock project data
    return {
        "data": {
            "id": project_id,
            "type": "project",
            "attributes": {
                "name": "MockProject"
            }
        }
    }


@app.get("/api/projects/v1beta2")
async def list_projects(filter: str = Query(None)):
    # Parse filter for project name
    project_id = str(uuid.uuid4())
    return {
        "data": [{
            "id": project_id,
            "type": "project",
            "attributes": {
                "name": "MockProject"
            }
        }]
    }


# QIR module creation endpoint
@app.post("/api/qir/v1beta/")
async def create_qir_module(module: dict):
    global createdQIRModules
    module_id = str(uuid.uuid4())
    createdQIRModules[module_id] = module.get("data",
                                              {}).get("attributes",
                                                      {}).get("contents", "")
    # Return response with module ID
    return {
        "data": {
            "id": module_id,
            "type": "qir",
            "attributes": module.get("data", {}).get("attributes", {})
        }
    }


# Job creation endpoint
@app.post("/api/jobs/v1beta3/")
async def create_job(job: dict):
    global createdQIRModules, createdJobs

    job_id = str(uuid.uuid4())
    job_name = job.get("data", {}).get("attributes", {}).get("name", "")
    items = job.get("data", {}).get("attributes", {}).get("definition",
                                                          {}).get("items", [])
    if not items:
        raise HTTPException(status_code=400,
                            detail="No items in job definition")
    shots = items[0].get("n_shots", 1000)
    program_id = items[0].get("program_id", "")
    program = createdQIRModules[program_id]
    decoded = base64.b64decode(program)
    m = llvm.module.parse_bitcode(decoded)
    mstr = str(m)
    assert ('entry_point' in mstr)

    # Get the function, number of qubits, and kernel name
    function = getKernelFunction(m)
    if function == None:
        raise Exception("Could not find kernel function")
    numQubitsRequired = getNumRequiredQubits(function)
    kernelFunctionName = function.name

    print("Kernel name = ", kernelFunctionName)
    print("Requires {} qubits".format(numQubitsRequired))

    # JIT Compile and get Function Pointer
    engine.add_module(m)
    engine.finalize_object()
    engine.run_static_constructors()
    funcPtr = engine.get_function_address(kernelFunctionName)
    kernel = ctypes.CFUNCTYPE(None)(funcPtr)

    # Invoke the Kernel
    cudaq.testing.toggleDynamicQubitManagement()
    qubits, context = cudaq.testing.initialize(numQubitsRequired, shots)
    kernel()
    results = cudaq.testing.finalize(qubits, context)
    results.dump()
    createdJobs[job_id] = (job_name, results)

    engine.remove_module(m)

    return {
        "data": {
            "id": job_id,
            "type": "job",
            "attributes": {
                "name": job_name,
                "status": {
                    "status": "QUEUED"
                }
            }
        }
    }


# Update job status retrieval
@app.get("/api/jobs/v1beta3/{job_id}")
async def get_job_status(job_id: str):
    global countJobGetRequests

    # Simulate asynchronous execution
    if countJobGetRequests < 3:
        countJobGetRequests += 1
        return {
            "data": {
                "id": job_id,
                "attributes": {
                    "status": {
                        "status": "RUNNING"
                    }
                }
            }
        }

    # Job completed
    countJobGetRequests = 0
    job_name, counts = createdJobs[job_id]
    result_id = str(uuid.uuid4())

    return {
        "data": {
            "id": job_id,
            "attributes": {
                "status": {
                    "status": "COMPLETED"
                },
                "definition": {
                    "items": [{
                        "result_id": result_id
                    }]
                }
            }
        }
    }


# Add results retrieval endpoint
@app.get("/api/results/v1beta3/{result_id}")
async def get_results(result_id: str):
    global createdJobs
    # Find the job that produced this result
    # This is a simplified implementation, and may need to be updated
    for job_id, (name, counts) in createdJobs.items():
        # Format counts for Nexus API format
        formatted_counts = []
        for bits, count in counts.items():
            formatted_counts.append({"bitstring": bits, "count": count})

        return {
            "data": {
                "id": result_id,
                "type": "result",
                "attributes": {
                    "counts_formatted": formatted_counts
                }
            }
        }
    raise HTTPException(status_code=404, detail="Result not found")


def startServer(port):
    uvicorn.run(app, port=port, host='0.0.0.0', log_level="info")


if __name__ == '__main__':
    startServer(62440)
