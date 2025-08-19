# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import base64
import ctypes
import uuid
from typing import Any, List, Optional, Union

import cudaq
import uvicorn
from fastapi import FastAPI, Header, HTTPException
from llvmlite import binding as llvm
from pydantic import BaseModel

# Define the REST Server App
app = FastAPI()

# Define the port for the mock QCI server
port = 62449

# In-memory storage
createdJobs = {}
jobResults = {}
jobStatuses = {}
jobCountRequests = {}

# Count how many times the client has requested the Job
countJobGetRequests = 0
# Save how many qubits were needed for each test (emulates real backend)
numQubitsRequired = 0


# Mock QCI API Models
class JobRequest(BaseModel):
    code: str
    machine: str
    mappingReorderIdx: Optional[List[int]] = None
    name: str
    outputNames: Optional[Any] = None
    shots: int
    userData: Optional[Any] = None


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


# Here we expose a way to post jobs,
# Must have a Access Token, Job Program must be Adaptive Profile
# with entry_point tag
@app.post("/cudaq/v1/jobs")
async def postJob(job: JobRequest,
                  token: Union[str, None] = Header(alias="Authorization",
                                                   default=None)):
    global createdJobs, shots, numQubitsRequired

    if token == None:
        raise HTTPException(status_code(401), detail="Credentials not provided")

    print('Posting job with shots = ', job.shots)
    jobId = str(uuid.uuid4())
    jobName = job.name
    shots = job.shots
    program = job.code
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
    qubits, context = cudaq.testing.initialize(numQubitsRequired, job.shots)
    kernel()
    results = cudaq.testing.finalize(qubits, context)
    results.dump()
    createdJobs[jobId] = (jobName, results)

    engine.remove_module(m)

    # Job "created", return the id
    return {"id": jobId, "jobs": {"status": "running"}}


@app.get("/cudaq/v1/jobs/{job_id}")
async def get_job_status(job_id: str):
    global createdJobs, createdResults, port
    if job_id not in createdJobs:
        raise HTTPException(status_code=404, detail="Job not found")

    # Simulate job processing time
    jobCountRequests[job_id] = jobCountRequests.get(job_id, 0) + 1

    # After a few requests, mark the job as completed
    if jobCountRequests[job_id] >= 3:
        jobStatuses[job_id] = "completed"

        # Return completed job with result URL
        return {
            "id": job_id,
            "status": "completed",
            "exited": True,
            "resultUrl": f"http://localhost:{port}/cudaq/v1/results/{job_id}"
        }

    # Job still running
    return {"id": job_id, "status": "running", "exited": False}


@app.get("/cudaq/v1/results/{job_id}")
async def get_job_results(job_id: str):
    global createdJobs, jobResults, numQubitsRequired
    if job_id not in createdJobs:
        raise HTTPException(status_code=404, detail="Job not found")

    if job_id not in jobResults:
        # Prepare and store results
        _, counts = createdJobs[job_id]

        # Convert counts to measurements which is a list of length shots with the results of each shot
        measurements = []
        for r in counts:
            for _ in range(counts[r]):
                measurements.append([int(bit) for bit in r])

        # Create a list of indices like - "index":[["r00000",0],["r00001",1]]
        indices = [[f"r{i:05d}", i] for i in range(len(counts))]

        jobResults[job_id] = {
            "id": str(uuid.uuid4()),
            "job": {
                "id": job_id
            },
            "measurements": measurements,
            "index": indices
        }

    return jobResults[job_id]


def startServer(port=port):
    """Start the mock QCI server"""
    uvicorn.run(app, port=port, host='0.0.0.0', log_level="info")


if __name__ == '__main__':
    startServer()
