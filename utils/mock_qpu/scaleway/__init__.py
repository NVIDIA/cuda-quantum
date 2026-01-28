# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq
import uuid, base64, ctypes

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llvmlite import binding as llvm

app = FastAPI()


class CreateJobRequest(BaseModel):
    model_id: str
    parameters: str


class CreateSessionRequest(BaseModel):
    platform_id: str
    name: str


class CreateModelRequest(BaseModel):
    payload: str


class Job(BaseModel):
    name: str
    id: str
    model_id: str
    status: str
    session_id: str
    parameters: str


class Session(BaseModel):
    name: str
    id: str
    platform_id: str


class Platform(BaseModel):
    name: str
    id: str


class Model(BaseModel):
    id: str
    payload: str


class Database:
    jobs: dict = {}
    platforms: dict = {}
    job_results: dict = {}
    sessions: dict = {}
    models: dict = {}


database = Database()
id = "b77a0dba-dc62-n069-83a1-cld32ac77e4e"
database.platforms[id] = Platform(id=id, name="EMU-CUDAQ-H100")

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
                str(a)
                .split(f'required_num_qubits"=')[-1]
                .split(" ")[0]
                .replace('"', "")
                .replace("'", "")
            )
        elif "requiredQubits" in str(a):
            return int(
                str(a)
                .split(f'requiredQubits"=')[-1]
                .split(" ")[0]
                .replace('"', "")
                .replace("'", "")
            )


def getKernelFunction(module):
    for f in module.functions:
        if not f.is_declaration:
            return f
    return None


def getNumRequiredQubits(function):
    for a in function.attributes:
        if "requiredQubits" in str(a):
            return int(
                str(a)
                .split('requiredQubits"=')[-1]
                .split(" ")[0]
                .replace('"', "")
                .replace("'", "")
            )


@app.get("/platforms")
async def listPlatforms():
    return (
        [
            {"id": platform.id, "name": platform.name}
            for platform in database.platforms.values()
        ],
        201,
    )


@app.get("/platforms/{platformId}")
async def getPlatform(platformId: str):
    platform = database.platforms.get(platformId)

    if not platform:
        raise HTTPException(status_code=404, detail="Platform not found")

    return ({"id": platform.id, "name": platform.name}, 201)


@app.post("/sessions")
async def createSession(request: CreateSessionRequest):
    session = Session(
        name=request.name, id=str(uuid.uuid4()), platform_id=request.platform_id
    )
    database.sessions[session.id] = session
    return ({"session_id": session.id}, 201)


@app.post("/models")
async def createModel(request: CreateModelRequest):
    model = Model(id=str(uuid.uuid4()), payload=request.payload)
    database.models[model.id] = model
    return ({"id": model.id, "payload": model.payload}, 201)


@app.post("/jobs")
async def createJob(request: CreateJobRequest):
    job = Job(
        name=request.name,
        id=str(uuid.uuid4()),
        model_id=request.model_id,
        status="created",
        session_id=None,
        parameters=request.parameters,
    )
    database.jobs[job.id] = job

    model = database.models.get(job.model_id)

    print("Posting job with name = ", job.name, job.count)
    newId = str(uuid.uuid4())
    program = model.payload
    decoded = base64.b64decode(program)
    m = llvm.module.parse_bitcode(decoded)
    mstr = str(m)
    assert "entry_point" in mstr

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

    engine.remove_module(m)

    # Job "created", return the id
    return ({"job_token": newId}, 201)


# Retrieve the job, simulate having to wait by counting to 3
# until we return the job results
@app.get("/jobs/{jobId}")
async def getJob(jobId: str):
    global countJobGetRequests

    # Simulate asynchronous execution
    if countJobGetRequests < 3:
        countJobGetRequests += 1
        return ({"job_id": jobId, "status": "running"}, 201)

    countJobGetRequests = 0
    # name, counts = createdJobs[jobId]
    retData = []
    for bits, count in counts.items():
        retData += [bits] * count

    # The simulators don't implement result recording features yet, so we have
    # to mark these results specially (MOCK_SERVER_RESULTS) in order to allow
    # downstream code to recognize that this isn't from a true QPU.
    res = ({"status": "done", "results": {"MOCK_SERVER_RESULTS": retData}}, 201)
    return res
