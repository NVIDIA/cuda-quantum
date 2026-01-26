# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
from typing import Union

import base64
import ctypes
import cudaq
import uuid
from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import JSONResponse
from llvmlite import binding as llvm
from pydantic import BaseModel
import json
from typing import Any, Dict

# Define the REST Server App
app = FastAPI()


# Jobs look like the following type
class Task(BaseModel):
    task_id: str
    program: str
    config: str


class TaskBody(BaseModel):
    tasks: list[Task]


class AuthModel(BaseModel):
    email: str
    password: str


class TaskIdRequest(BaseModel):
    qpu_id: str
    task_count: int
    tag: str


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
@app.post("/{deviceId}/tasks/submit")
async def postJob(data: Dict[str, Any],
                  authentication_token: str = Header(...),
                  content_type: str = Header(...)):
    global createdJobs, shots
    if authentication_token != "fake_auth_token":
        raise HTTPException(status_code=403, detail="Permission denied")
    tasks = TaskBody(tasks=data["tasks"])
    for task in tasks.tasks:
        newId = task.task_id
        program = task.program
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
        qubits, context = cudaq.testing.initialize(numQubitsRequired, 1000)
        kernel()
        results = cudaq.testing.finalize(qubits, context)
        results.dump()
        createdJobs[newId] = (task.task_id, results)

    engine.remove_module(m)

    # Job "created", return the id
    return {"job": newId}


# Retrieve the job, simulate having to wait by counting to 3
# until we return the job results
@app.get("/{deviceId}/tasks/{jobId}/all_info")
async def getJob(jobId: str):
    global countJobGetRequests, createdJobs, shots

    countJobGetRequests = 0
    name, counts = createdJobs[jobId]
    retData = {}
    for bits, count in counts.items():
        retData[str(bits)] = count

    return {"results": retData}


@app.post("/tasks")
async def getJob(n=1):
    return [uuid.uuid4() for _ in range(n)]


@app.post("/{deviceId}/tasks")
async def reserveJobId(request: TaskIdRequest):
    n = request.task_count
    return [uuid.uuid4() for _ in range(n)]


@app.get("/admin/qpu")
async def qetQpu(authentication_token: str = Header(...)):

    if authentication_token != "fake_auth_token":
        raise HTTPException(status_code=403, detail="Permission denied")

    data = {
        "items": [{
            "active": True,
            "created_at": "2024-04-09T14:24:50.918020+00:00",
            "created_by": "11111111-1111-1111-1111-111111111111",
            "feature_set": {
                "always_on": True,
                "qubit_count": 8,
                "simulator": True
            },
            "generation": -1,
            "id": "qpu:uk:-1:1234567890",
            "name": "OQC Mock Server",
            "region": "uk",
            "status": "ACTIVE",
            "updated_at": "2024-07-29T14:54:46.948951+00:00",
            "updated_by": "11111111-1111-1111-1111-111111111111",
            "url": "http://localhost:62442/1234567890"
        }],
        "page": 1,
        "per_page": 10,
        "total": 2
    }

    return JSONResponse(content=data)
