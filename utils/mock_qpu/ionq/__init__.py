# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq
from fastapi import FastAPI, HTTPException, Header
from typing import Union
import uvicorn, uuid, base64, ctypes
from pydantic import BaseModel
from llvmlite import binding as llvm

# Define the REST Server App
app = FastAPI()


class Input(BaseModel):
    format: str
    data: str


# Jobs look like the following type
class Job(BaseModel):
    target: str
    qubits: str
    shots: int
    input: Input


# Keep track of Job Ids to their Names
createdJobs = {}

# Could how many times the client has requested the Job
countJobGetRequests = 0

# Save how many qubits were needed for each test (emulates real backend)
numQubitsRequired = 0

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
@app.post("/login")
async def login(token: Union[str, None] = Header(alias="Authorization",
                                                 default=None)):
    if token == None:
        raise HTTPException(status_code(401), detail="Credentials not provided")
    return {"id-token": "hello", "refresh-token": "refreshToken"}


# Here we expose a way to post jobs,
# Must have a Access Token, Job Program must be Adaptive Profile
# with entry_point tag
@app.post("/v0.3/jobs")
async def postJob(job: Job,
                  token: Union[str, None] = Header(alias="Authorization",
                                                   default=None)):
    global createdJobs, shots, numQubitsRequired

    if token == None:
        raise HTTPException(status_code(401), detail="Credentials not provided")

    print('Posting job with shots = ', job.shots)
    newId = str(uuid.uuid4())
    shots = job.shots
    program = job.input.data
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
    qubits, context = cudaq.testing.initialize(numQubitsRequired, job.shots)
    kernel()
    results = cudaq.testing.finalize(qubits, context)
    results.dump()
    createdJobs[newId] = results

    engine.remove_module(m)

    # Job "created", return the id
    return {"id": newId, "jobs": {"status": "running"}}


# Retrieve the job, simulate having to wait by counting to 3
# until we return the job results
@app.get("/v0.3/jobs")
async def getJob(id: str):
    global countJobGetRequests, createdJobs, numQubitsRequired

    # Simulate asynchronous execution
    if countJobGetRequests < 3:
        countJobGetRequests += 1
        return {"jobs": [{"status": "running"}]}

    countJobGetRequests = 0
    res = {
        "jobs": [{
            "status": "completed",
            "qubits": numQubitsRequired,
            "results_url": "/v0.3/jobs/{}/results".format(id)
        }]
    }
    return res


@app.get("/v0.3/jobs/{jobId}/results")
async def getResults(jobId: str):
    global countJobGetRequests, createdJobs

    counts = createdJobs[jobId]
    counts.dump()
    retData = {}
    N = 0
    for bits, count in counts.items():
        N += count
    # Note, the real IonQ backend reverses the bitstring relative to what the
    # simulator does, so flip the bitstring with [::-1]. Also convert
    # to decimal to match the real IonQ backend.
    for bits, count in counts.items():
        retData[str(int(bits[::-1], 2))] = float(count / N)

    res = retData
    return res


def startServer(port):
    uvicorn.run(app, port=port, host='0.0.0.0', log_level="info")


if __name__ == '__main__':
    startServer(62455)
