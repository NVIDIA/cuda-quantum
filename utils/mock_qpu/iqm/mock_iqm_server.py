# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
import cudaq

import asyncio
import uuid
from fastapi import FastAPI, Request, HTTPException, Header
from typing import Optional, Union
import uvicorn, uuid, json, base64, ctypes
from pydantic import BaseModel, Field

# Use IQM Client Tools to verify data structures
import iqm_client


# Testing consts
good_access_token = "Bearer good_access_token"
server_qpu_architecture = "Apollo"
operations = []  # TBA
qubits = []  # TBA
qubit_connectivity = []  # TBA


# Define the REST Server App
app = FastAPI()


# Keep job artifacts
class Job(BaseModel):
    """Store Job stuff"""

    id: str
    status: iqm_client.Status
    result: Optional[iqm_client.RunResult] = None
    counts: Optional[dict[str, int]] = None
    metadata: iqm_client.Metadata


# New job created response
class PostJobsResponse(BaseModel):
    """POST /jobs response"""

    id: str


# Jobs storage
createdJobs: dict[str, Job] = {}


def generate_measurement_strings(n, bs=""):
    if n - 1:
        yield from generate_measurement_strings(n - 1, bs + "0")
        yield from generate_measurement_strings(n - 1, bs + "1")
    else:
        yield bs + "0"
        yield bs + "1"


async def compile_and_submit_job(job: Job):
    """Anaylze measurements and construct corresponding counts"""
    request = job.metadata.request
    circuits = request.circuits

    if len(circuits) != 1:
        job.status = iqm_client.Status.FAILED
        job.result = iqm_client.RunResult(
            status=job.status,
            metadata=job.metadata,
            message="Exactly one circuit must be provided, got {}".format(
                len(circuits)
            ),
        )
        createdJobs[job.id] = job
        return

    circuit = circuits[0]
    measurements = [
        instruction
        for instruction in circuit.instructions
        if instruction.name == "measurement"
    ]
    if len(measurements) == 0:
        job.status = iqm_client.Status.FAILED
        job.result = iqm_client.RunResult(
            status=job.status, metadata=job.metadata, message="Circuit contains no measurements"
        )
        createdJobs[job.id] = job
        return

    # check some connectivity
    qubit_pairs = [
        instruction.qubits
        for instruction in circuit.instructions
        if len(instruction.qubits) == 2
    ]
    if ("QB2", "QB3") in qubit_pairs or ("QB3", "QB2") in qubit_pairs:
        job.status = iqm_client.Status.FAILED
        job.result = iqm_client.RunResult(
            status=job.status,
            metadata=job.metadata,
            message="Some circuits in the batch have gates between uncoupled qubits:",
        )
        createdJobs[job.id] = job
        return

    # assume there is only one measurement at the end of the circuit
    measured_qubits = []
    for measurement in measurements:
        for qubit in measurement.qubits:
            measured_qubits.append(qubit)
    measured_qubits = list(set(measured_qubits))

    # populate counts according to amount of qubits in each measurement
    qubits_in_measurement = len(measured_qubits)
    job.counts = {}
    for measurement_string in generate_measurement_strings(qubits_in_measurement):
        job.counts[measurement_string] = 1

    job.status = iqm_client.Status.READY
    job.result = iqm_client.RunResult(status=job.status, metadata=job.metadata)
    createdJobs[job.id] = job


@app.get("/quantum-architecture")
async def get_quantum_architecture(request: Request) -> iqm_client.QuantumArchitecture:
    """Get the quantum architecture"""

    access_token = request.headers.get("Authorization")
    if access_token != good_access_token:
        raise HTTPException(401)

    return iqm_client.QuantumArchitecture(
        quantum_architecture=iqm_client.QuantumArchitectureSpecification(
            name=server_qpu_architecture,
            operations=operations,
            qubits=qubits,
            qubit_connectivity=qubit_connectivity,
        )
    )


@app.post("/jobs")
async def post_jobs(
    job_request: iqm_client.RunRequest, request: Request
) -> PostJobsResponse:
    """Register a new job and start execution"""

    access_token = request.headers.get("Authorization")
    if access_token != good_access_token:
        raise HTTPException(401)

    metadata = iqm_client.Metadata(request=job_request)
    new_job_id = str(uuid.uuid4())
    new_job = Job(
        id=new_job_id,
        status=iqm_client.Status.PENDING_COMPILATION,
        request=job_request,
        metadata=metadata,
    )
    createdJobs[new_job_id] = new_job

    # start async compilation and execution
    asyncio.create_task(compile_and_submit_job(new_job))
    await asyncio.sleep(0.0)

    return PostJobsResponse(id=new_job_id)


@app.get("/jobs/{job_id}/status")
async def get_jobs_status(job_id: str, request: Request) -> iqm_client.Status:
    """Get the status of a job"""

    access_token = request.headers.get("Authorization")
    if access_token != good_access_token:
        raise HTTPException(401)

    if job_id not in createdJobs:
        raise HTTPException(404)

    return createdJobs[job_id].status


@app.get("/jobs/{job_id}/results/counts")
async def get_jobs(job_id: str, request: Request):
    """Get the result of a job"""
    access_token = request.headers.get("Authorization")
    if access_token != good_access_token:
        raise HTTPException(401)

    if job_id not in createdJobs:
        raise HTTPException(404)

    job = createdJobs[job_id]

    # TODO: return the actual counts, check the requested measurements
    results = {
        "status": job.status,
        "message": job.result.message if job.result and job.result.message else None,
        "counts": job.counts,
    }

    return results


def startServer(port):
    uvicorn.run(app, port=port, host="0.0.0.0", log_level="debug")


if __name__ == "__main__":
    startServer(9100)
