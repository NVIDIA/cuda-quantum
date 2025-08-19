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

# Keep track of the result ID to the job it was created from
createdResults = {}

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
@app.post("/auth/login")
async def login(token: Union[str, None] = Header(alias="Authorization",
                                                 default=None)):
    if 'token' == None:
        raise HTTPException(status_code(401), detail="Credentials not provided")
    return {"id-token": "hello", "refresh-token": "refreshToken"}


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
# Job Program must be Adaptive Profile with entry_point tag
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


# Retrieve the job, simulate having to wait by counting to 3 until we return the job results
@app.get("/api/jobs/v1beta3/{job_id}")
async def get_job_status(job_id: str):
    global countJobGetRequests, createdResults

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

    result_id = str(uuid.uuid4())
    createdResults[result_id] = job_id

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
    global createdJobs, createdResults
    # Find the job that produced this result
    # This is a simplified implementation, and may need to be updated
    job_id = createdResults.get(result_id)
    if not job_id:
        raise HTTPException(status_code=404, detail="Result not found")

    _, counts = createdJobs[job_id]

    # Get the exact length of the first bitstring
    bit_length = len(list(counts.items())[0][0]) if counts else 0

    # Format counts for Nexus API format
    formatted_counts = []
    outcome_array = []
    for bits, count in counts.items():
        formatted_counts.append({"bitstring": bits, "count": count})
        outcome_bytes = []
        reverse = bits[::-1]
        # Padding bits to ensure we have a multiple of 8
        if len(reverse) % 8 != 0:
            reverse += '0' * (8 - len(reverse) % 8)
        for i in range(0, bit_length, 8):
            byte_value = int(reverse[i:i + 8], 2)
            outcome_bytes.append(byte_value)
        for _ in range(count):
            outcome_array.append(outcome_bytes)

    shots_data = {"width": bit_length, "array": outcome_array}
    # Create properly formatted register names (r00000, r00001, etc.)
    bits_metadata = []
    for i in range(bit_length):
        reg_idx = bit_length - i - 1  # Reverse order to match Quantinuum format
        reg_name = f"r{reg_idx:05d}"
        bits_metadata.append([reg_name, [0]])

    return {
        "data": {
            "id": result_id,
            "type": "result",
            "attributes": {
                "bits": bits_metadata,
                "shots": shots_data,
                "counts": [],
                "counts_formatted": formatted_counts
            }
        }
    }


def startServer(port):
    uvicorn.run(app, port=port, host='0.0.0.0', log_level="info")


if __name__ == '__main__':
    startServer(62440)
