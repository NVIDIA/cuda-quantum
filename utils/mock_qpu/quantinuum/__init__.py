# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq
from fastapi import FastAPI, HTTPException, Header, Query
from fastapi.responses import JSONResponse
from typing import Union
import uuid, base64, ctypes
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

# Keep track of created decoder configurations
createdDecoderConfigs = {}

llvm.initialize()
llvm.initialize_native_target()
llvm.initialize_native_asmprinter()
target = llvm.Target.from_default_triple()
targetMachine = target.create_target_machine()
backing_mod = llvm.parse_assembly("")
engine = llvm.create_mcjit_compiler(backing_mod, targetMachine)
# Verbose logging for debugging
verbose = False


def getNumRequiredQubits(function):
    for a in function.attributes:
        if "required_num_qubits" in str(a):
            return int(
                str(a).split(f'required_num_qubits\"=')[-1].split(" ")
                [0].replace("\"", "").replace("'", ""))
        elif "requiredQubits" in str(a):
            return int(
                str(a).split(f'requiredQubits\"=')[-1].split(" ")[0].replace(
                    "\"", "").replace("'", ""))


def getNumRequiredResults(function):
    for a in function.attributes:
        if "required_num_results" in str(a):
            return int(
                str(a).split(f'required_num_results\"=')[-1].split(" ")
                [0].replace("\"", "").replace("'", ""))
        elif "requiredResults" in str(a):
            return int(
                str(a).split(f'requiredResults\"=')[-1].split(" ")[0].replace(
                    "\"", "").replace("'", ""))


def getKernelFunction(module):
    for f in module.functions:
        if not f.is_declaration:
            return f
    return None


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

    device_name = job.get("data",
                          {}).get("attributes",
                                  {}).get("definition",
                                          {}).get("backend_config",
                                                  {}).get("device_name", "")
    if verbose:
        print("Job data =", job)
        print("Device name =", device_name)
    # If device name starts with "Helios", we assume it's an NG device
    is_ng_device = device_name.startswith("Helios")

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
    if verbose:
        print("Code")
        print(mstr)

    # If any decoder configuration ID is provided, check it exists
    decoder_config_id = job.get("data",
                                {}).get("attributes",
                                        {}).get("definition",
                                                {}).get("gpu_decoder_config_id",
                                                        None)
    if decoder_config_id is not None:
        if verbose:
            print("Decoder config ID provided:", decoder_config_id)
        if decoder_config_id not in createdDecoderConfigs:
            raise HTTPException(status_code=400,
                                detail="Invalid decoder configuration ID")

    # Get the function, number of qubits, and kernel name
    function = getKernelFunction(m)
    if function == None:
        raise Exception("Could not find kernel function")
    numQubitsRequired = getNumRequiredQubits(function)
    numResultsRequired = getNumRequiredResults(function)
    kernelFunctionName = function.name

    if verbose:
        print("Kernel name = ", kernelFunctionName)
        print("Requires {} qubits".format(numQubitsRequired))
        print("Requires {} results".format(numResultsRequired))

    # JIT Compile and get Function Pointer
    engine.add_module(m)
    engine.finalize_object()
    engine.run_static_constructors()
    funcPtr = engine.get_function_address(kernelFunctionName)
    kernel = ctypes.CFUNCTYPE(None)(funcPtr)

    # Clear any leftover log from previous jobs
    cudaq.testing.getAndClearOutputLog()

    # Invoke the Kernel
    if is_ng_device:
        qir_log = f"HEADER\tschema_id\tlabeled\nHEADER\tschema_version\t1.0\nSTART\nMETADATA\tentry_point\nMETADATA\tqir_profiles\tadaptive_profile\nMETADATA\trequired_num_qubits\t{numQubitsRequired}\nMETADATA\trequired_num_results\t{numResultsRequired}\n"

        for i in range(shots):
            cudaq.testing.toggleDynamicQubitManagement()
            qubits, context = cudaq.testing.initialize(numQubitsRequired, 1,
                                                       "run")
            kernel()
            _ = cudaq.testing.finalize(qubits, context)

            shot_log = cudaq.testing.getAndClearOutputLog()
            if i > 0:
                qir_log += "START\n"
            qir_log += shot_log
            qir_log += "END\t0\n"

        createdJobs[job_id] = (job_name, qir_log)
    else:
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

    is_qsys_job = isinstance(createdJobs[job_id][1], str)

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
                        "result_id": result_id,
                        "result_type": "QSYS" if is_qsys_job else "PYTKET",
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


# NG device results retrieval endpoint (`qsys_results`)
@app.get("/api/qsys_results/v1beta/{result_id}")
async def get_results(result_id: str, version: int):
    # Version can only be 3 (default)
    if version not in [3]:
        raise HTTPException(status_code=400, detail="Invalid version")
    global createdJobs, createdResults
    # Find the job that produced this result
    # This is a simplified implementation, and may need to be updated
    job_id = createdResults.get(result_id)
    if not job_id:
        raise HTTPException(status_code=404, detail="Result not found")

    _, qir_log = createdJobs[job_id]

    if verbose:
        print("QIR output log:")
        print(qir_log)

    return {
        "data": {
            "id": result_id,
            "attributes": {
                "results": qir_log
            },
            "relationships": {
                "program": {
                    "data": {
                        "type": "qir"
                    }
                }
            }
        }
    }


# Endpoint to upload decoder configuration
@app.post("/api/gpu_decoder_configs/v1beta/")
async def create_decoder_config(job: dict):
    global createdDecoderConfigs
    config_id = str(uuid.uuid4())
    config_base64 = job.get("data", {}).get("attributes",
                                            {}).get("contents", "")
    # Decode the base64 string
    try:
        config_str = base64.b64decode(config_base64,
                                      validate=True).decode('utf-8')
    except Exception as e:
        raise HTTPException(status_code=400,
                            detail="Invalid base64 encoding") from e

    # Check that the configuration is valid YAML
    import yaml
    try:
        yaml.safe_load(config_str)
    except yaml.YAMLError as e:
        raise HTTPException(status_code=400,
                            detail="Invalid YAML format") from e

    createdDecoderConfigs[config_id] = config_str
    # Return response with module ID
    return {
        "data": {
            "id": config_id,
            "type": "gpu_decoder_config",
            "attributes": {
                "contents": config_base64
            }
        }
    }
