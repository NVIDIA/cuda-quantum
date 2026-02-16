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

from qio.core import (
    QuantumComputationModel,
    QuantumComputationParameters,
    QuantumProgramResult,
)

class CreateJobRequest(BaseModel):
    name: str
    model_id: str
    session_id: str
    parameters: str


class CreateSessionRequest(BaseModel):
    platform_id: str
    name: str
    max_duration: str
    max_idle_duration: str


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
    max_duration: str
    max_idle_duration: str


class Platform(BaseModel):
    name: str
    id: str
    availability: str
    backend_name: str
    provider_name: str
    version: str
    type: str
    technology: str
    max_qubit_count: int
    max_shot_count: int
    max_circuit_count: int
    metadata: str


class Model(BaseModel):
    id: str
    payload: str


app = FastAPI()

class Database:
    jobs: dict = {}
    platforms: dict = {}
    job_results: dict = {}
    sessions: dict = {}
    models: dict = {}


countJobGetRequests = 0

_BASE_PATH = "/qaas/v1alpha1"
_FAKE_PLATFORM_ID = "b77a0dba-dc62-n069-83a1-cld32ac77e4e"

database = Database()
database.platforms[_FAKE_PLATFORM_ID] = Platform(
    id=_FAKE_PLATFORM_ID,
    name="EMU-CUDAQ-FAKE",
    provider_name="nvidia",
    backend_name="cudaq",
    version="0.0",
    availability="available",
    max_qubit_count=20,
    max_circuit_count=1,
    max_shot_count=10000,
    metadata="",
    technology="general_purpose",
    type="simulator",
)


# @cudaq.kernel
# def _bell_kernel():
#     qubits = cudaq.qvector(2)
#     h(qubits[0])
#     x.ctrl(qubits[0], qubits[1])
#     mz(qubits)


def _run_job(job: Job):
    # Try to retrieve provided shot counts
    model = QuantumComputationModel.from_json_str(job.payload)
    params = QuantumComputationParameters.from_json_str(job.parameters)

    # try:
        # shot_count = json.loads(job.parameters)["shots"]
    shot_count = params.shots
    # except Exception as e:
        # shot_count = 100

    kernel = model.programs[0].to_cudaq_kernel()

    # Run a bell state as mock execution
    sample_result = cudaq.sample(kernel, shots_count=shot_count)
    job.status = "completed"

    # Simplified qio result
    # result = json.dumps(
    #     {
    #         "serialization": json.dumps(sample_result.serialize()),
    #         "serialization_format": 3,  # CUDA-Q Sample Result
    #         "compression_format": 1,  # No compression
    #     }
    # )

    result = QuantumProgramResult.from_cudaq_sample_result(result).to_json_str()

    result = JobResult(id=str(uuid.uuid4()), job_id=job.id, result=result, url="")
    database.job_results[result.id] = result


@app.get(_BASE_PATH + "/platforms")
async def listPlatforms(name: str | None = None):
    if name:
        filtered_plts = list(
            filter(lambda p: p.name == name, database.platforms.values())
        )
        platforms = [platform.model_dump() for platform in filtered_plts]
    else:
        platforms = [platform.model_dump() for platform in database.platforms.values()]

    return {"platforms": platforms, "total_count": len(platforms)}


@app.get(_BASE_PATH + "/platforms/{platformId}")
async def getPlatform(platformId: str):
    platform = database.platforms.get(platformId)

    if not platform:
        raise HTTPException(status_code=404, detail="Platform not found")

    return platform.model_dump()


@app.post(_BASE_PATH + "/sessions")
async def createSession(request: CreateSessionRequest):
    session = Session(
        name=request.name,
        id=str(uuid.uuid4()),
        platform_id=request.platform_id,
        max_duration=request.max_duration,
        max_idle_duration=request.max_idle_duration,
    )
    database.sessions[session.id] = session
    return session.model_dump()


@app.get(_BASE_PATH + "/sessions/{sessionId}")
async def getSession(sessionId: str):
    session = database.sessions.get(sessionId)

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return session.model_dump()


@app.post(_BASE_PATH + "/models")
async def createModel(request: CreateModelRequest):
    model = Model(id=str(uuid.uuid4()), payload=request.payload)
    database.models[model.id] = model
    return model.model_dump()


@app.post(_BASE_PATH + "/jobs")
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

    return job.model_dump()


# Retrieve the job, simulate having to wait by counting to 3
# until we return the job results
@app.get(_BASE_PATH + "/jobs/{jobId}")
async def getJob(jobId: str):
    global countJobGetRequests

    job = database.jobs.get(jobId)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    job.status = "running"

    # Simulate asynchronous execution
    if countJobGetRequests < 3:
        countJobGetRequests += 1
        return job.model_dump()

    _run_job(job)

    countJobGetRequests = 0

    return job.model_dump()


@app.get(_BASE_PATH + "/jobs/{jobId}/results")
async def listJobResults(jobId: str):
    if not database.jobs.get(jobId):
        raise HTTPException(status_code=404, detail="Job not found")

    results = [
        result.model_dump()
        for result in list(
            filter(lambda r: r.job_id == jobId,
                   database.job_results.values()
            )
        )
    ]

    return {"job_results": results, "total_count": len(results)}
