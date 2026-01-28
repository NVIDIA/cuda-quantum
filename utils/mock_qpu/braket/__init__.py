# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq
from fastapi import FastAPI, HTTPException, Header
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


# Here we test that the login endpoint works
@app.post("/login")
async def login(token: Union[str, None] = Header(alias="Authorization",
                                                 default=None)):
    if 'token' == None:
        raise HTTPException(status_code(401), detail="Credentials not provided")
    return {"id_token": "hello", "refresh_token": "refreshToken"}


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
    return ({"job_token": newId}, 201)


# Retrieve the job, simulate having to wait by counting to 3
# until we return the job results
@app.get("/job/{jobId}")
async def getJob(jobId: str):
    global countJobGetRequests, createdJobs, shots

    # Simulate asynchronous execution
    if countJobGetRequests < 3:
        countJobGetRequests += 1
        return ({"status": "executing"}, 201)

    countJobGetRequests = 0
    name, counts = createdJobs[jobId]
    retData = []
    for bits, count in counts.items():
        retData += [bits] * count

    # The simulators don't implement result recording features yet, so we have
    # to mark these results specially (MOCK_SERVER_RESULTS) in order to allow
    # downstream code to recognize that this isn't from a true QPU.
    res = ({"status": "done", "results": {"MOCK_SERVER_RESULTS": retData}}, 201)
    return res
