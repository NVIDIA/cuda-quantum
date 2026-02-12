# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import uuid
import cudaq
import json

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()


class CreateJobRequest(BaseModel):
    model_id: str
    session_id: str
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


class JobResult(BaseModel):
    id: str
    job_id: str
    url: str
    result: str


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


countJobGetRequests = 0

database = Database()
_FAKE_PLATFORM_ID = "b77a0dba-dc62-n069-83a1-cld32ac77e4e"
database.platforms[_FAKE_PLATFORM_ID] = Platform(
    id=_FAKE_PLATFORM_ID, name="EMU-CUDAQ-H100"
)


@cudaq.kernel
def _bell_kernel():
    qubits = cudaq.qvector(2)
    h(qubits[0])
    x.ctrl(qubits[0], qubits[1])
    mz(qubits)


def _run_fake_job(job: Job):
    # Try to retrieve provided shot counts
    try:
        shot_count = json.loads(job.parameters)["shots"]
    except Exception as e:
        shot_count = 100

    # Run a bell state as mock execution
    sample_result = cudaq.sample(_bell_kernel, shots_count=shot_count)
    job.status = "completed"

    # Simplified qio result
    result = json.dumps(
        {
            "serialization": json.dumps(sample_result.serialize()),
            "serialization_format": 3,  # CUDA-Q Sample Result
            "compression_format": 1,  # No compression
        }
    )

    result = JobResult(id=str(uuid.uuid4()), job_id=job.id, result=result)
    database.job_results[result.id] = result


@app.get("/platforms")
async def listPlatforms():
    return (
        [platform.model_dump() for platform in database.platforms.values()],
        201,
    )


@app.get("/platforms/{platformId}")
async def getPlatform(platformId: str):
    platform = database.platforms.get(platformId)

    if not platform:
        raise HTTPException(status_code=404, detail="Platform not found")

    return (platform.model_dump(), 201)


@app.post("/sessions")
async def createSession(request: CreateSessionRequest):
    session = Session(
        name=request.name, id=str(uuid.uuid4()), platform_id=request.platform_id
    )
    database.sessions[session.id] = session
    return (session.model_dump(), 201)


@app.post("/models")
async def createModel(request: CreateModelRequest):
    model = Model(id=str(uuid.uuid4()), payload=request.payload)
    database.models[model.id] = model
    return (model.model_dump(), 201)


@app.post("/jobs")
async def createJob(request: CreateJobRequest):
    job = Job(
        name=request.name,
        id=str(uuid.uuid4()),
        model_id=request.model_id,
        status="waiting",
        session_id=request.session_id,
        parameters=request.parameters,
    )
    database.jobs[job.id] = job

    return (job.model_dump(), 201)


# Retrieve the job, simulate having to wait by counting to 3
# until we return the job results
@app.get("/jobs/{jobId}")
async def getJob(jobId: str):
    global countJobGetRequests

    job = database.jobs.get(jobId)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    job.status = "running"

    # Simulate asynchronous execution
    if countJobGetRequests < 3:
        countJobGetRequests += 1
        return (job.model_dump(), 201)

    _run_fake_job(job)

    countJobGetRequests = 0

    return (job.model_dump(), 201)


@app.get("/jobs/{jobId}/results")
async def listJobResults(jobId: str):
    if not database.jobs.get(jobId):
        raise HTTPException(status_code=404, detail="Job not found")

    return (
        [
            result.model_dump()
            for result in list(
                filter(lambda r: r.job_id == jobId, database.job_results.values())
            )
        ],
        201,
    )
