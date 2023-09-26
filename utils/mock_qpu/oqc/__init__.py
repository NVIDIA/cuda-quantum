# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
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
import uvicorn
from fastapi import FastAPI, HTTPException, Header
from llvmlite import binding as llvm
from pydantic import BaseModel

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
                    "\"", ""))


# Here we test that the login endpoint works
@app.post("/auth")
async def login(auth_info: AuthModel):
    return {"access_token": "auth_token"}


# Here we expose a way to post jobs,
# Must have a Access Token, Job Program must be Adaptive Profile
# with entry_point tag
@app.post("/tasks/submit")
async def postJob(
    tasks: Union[TaskBody, Task],
    # access_token: Union[str, None] = Header(alias="Authorization",default=None)
):
    global createdJobs, shots

    # if access_token == None:
    # raise HTTPException(status_code(401), detail="Credentials not provided")
    if isinstance(tasks, Task):
        tasks = TaskBody(tasks=[
            tasks,
        ])
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
        cudaq.testing.toggleBaseProfile()
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
@app.get("/tasks/{jobId}/results")
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


def startServer(port):
    uvicorn.run(app, port=port, host='0.0.0.0', log_level="info")


if __name__ == '__main__':
    startServer(62454)
