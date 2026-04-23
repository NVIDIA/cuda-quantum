# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import itertools
import random
import re
import uuid
from typing import Any, Optional

import uvicorn
from fastapi import FastAPI, Header, HTTPException, Path
from pydantic import BaseModel

app = FastAPI()


class Program(BaseModel):
    """Structured program payload for v2 API."""

    format: str
    data: str


class Job(BaseModel):
    """Data required to submit a quantum job (v2 API)."""

    program: Program
    shots: int
    deviceQrn: str
    name: Optional[str] = None
    tags: Optional[dict] = None


JOBS_MOCK_DB = {}
JOBS_MOCK_RESULTS = {}
# Testing toggle: when True, the next job submitted via POST /jobs is created
# with status FAILED. Consumed (reset to False) after use.
FAIL_NEXT_JOB = {"enabled": False}
# Testing counter: how many upcoming GET /jobs/{id}/result calls should return
# success=false (simulating the qbraid v2 race where status=COMPLETED before
# results are queryable). Decrements on each /result call until 0.
DELAY_RESULTS_COUNT = {"remaining": 0}
# Testing hook: when set, the next GET /jobs/{id}/result call raises the given
# HTTP status. Consumed (reset to None) after one call. Used to exercise the
# helper's 401/403/404/5xx handling paths.
FORCE_NEXT_RESULT_STATUS = {"code": None}


def count_qubits(qasm: str) -> int:
    """Extracts the number of qubits from an OpenQASM string."""
    pattern = r"qreg\s+\w+\[(\d+)\];"

    match = re.search(pattern, qasm)

    if match:
        return int(match.group(1))

    raise ValueError("No qreg declaration found in the OpenQASM string.")


def simulate_job(qasm: str, num_shots: int) -> dict[str, int]:
    """Simulates a quantum job by generating random measurement outcomes based on the circuit."""
    num_qubits = count_qubits(qasm)

    measured_qubits = []

    measure_pattern = r"measure\s+(\w+)\[(\d+)\]"
    measure_matches = re.findall(measure_pattern, qasm)

    hadamard_pattern = r"h\s+(\w+)\[(\d+)\]"
    hadamard_matches = re.findall(hadamard_pattern, qasm)

    superposition_qubits = set()
    for _, qubit_idx in hadamard_matches:
        superposition_qubits.add(int(qubit_idx))

    for _, qubit_idx in measure_matches:
        measured_qubits.append(int(qubit_idx))

    if not measured_qubits:
        measured_qubits = list(range(num_qubits))

    result = {}

    possible_states = []

    if measured_qubits:
        # Generate strings of the appropriate length for measured qubits
        # For superposition qubits, include both 0 and 1 outcomes
        for measured_qubit in measured_qubits:
            if measured_qubit in superposition_qubits:
                if not possible_states:
                    possible_states = ["0", "1"]
                else:
                    new_states = []
                    for state in possible_states:
                        new_states.append(state + "0")
                        new_states.append(state + "1")
                    possible_states = new_states
            else:
                if not possible_states:
                    possible_states = ["0"]
                else:
                    possible_states = [state + "0" for state in possible_states]

    if not possible_states:
        if superposition_qubits:
            possible_states = ["0", "1"]
        else:
            possible_states = ["0" * num_qubits]

    distribution = random.choices(possible_states, k=num_shots)
    result = {state: distribution.count(state) for state in set(distribution)}

    if (num_qubits == 2 and len(measured_qubits) == 1 and
            measured_qubits[0] == 0 and 0 in superposition_qubits):
        new_result = {}
        total_shots = num_shots
        half_shots = total_shots // 2

        new_result["00"] = random.randint(half_shots - half_shots // 4,
                                          half_shots + half_shots // 4)
        new_result["01"] = 0
        new_result["10"] = random.randint(half_shots - half_shots // 4,
                                          half_shots + half_shots // 4)
        new_result["11"] = 0

        remaining = total_shots - (new_result["00"] + new_result["10"])
        if remaining > 0:
            new_result["00"] += remaining

        result = {k: v for k, v in new_result.items() if v > 0}

    return result


def poll_job_status(job_id: str) -> dict[str, Any]:
    """Updates the status of a job and returns the updated job data."""
    if job_id not in JOBS_MOCK_DB:
        raise HTTPException(status_code=404, detail="Job not found")

    status = JOBS_MOCK_DB[job_id]["status"]

    status_transitions = {
        "INITIALIZING": "QUEUED",
        "QUEUED": "RUNNING",
        "RUNNING": "COMPLETED",
        "CANCELLING": "CANCELLED",
    }

    new_status = status_transitions.get(status, status)
    JOBS_MOCK_DB[job_id]["status"] = new_status

    return {"jobQrn": job_id, **JOBS_MOCK_DB[job_id]}


# v2 API: POST /jobs
@app.post("/jobs")
async def postJob(job: Job,
                  x_api_key: Optional[str] = Header(None, alias="X-API-KEY")):
    """Submit a quantum job for execution (v2 API)."""
    if x_api_key is None:
        raise HTTPException(status_code=401, detail="API key is required")

    newId = str(uuid.uuid4())

    # Test hook: fail this job immediately if the toggle was armed.
    if FAIL_NEXT_JOB["enabled"]:
        FAIL_NEXT_JOB["enabled"] = False
        job_data = {
            "status": "FAILED",
            "statusText": "Triggered failure for testing",
            **job.model_dump(),
        }
        JOBS_MOCK_DB[newId] = job_data
        return {"success": True, "data": {"jobQrn": newId, "status": "FAILED"}}

    # Extract QASM from the structured program payload
    counts = simulate_job(job.program.data, job.shots)

    job_data = {"status": "INITIALIZING", "statusText": "", **job.model_dump()}

    JOBS_MOCK_DB[newId] = job_data
    JOBS_MOCK_RESULTS[newId] = counts

    # v2 response: wrapped in success/data envelope
    return {
        "success": True,
        "data": {
            "jobQrn": newId,
            "status": "INITIALIZING"
        }
    }


# Test-only: arm a failure for the next submitted job.
@app.post("/test/fail_next")
async def armFailNext():
    FAIL_NEXT_JOB["enabled"] = True
    return {"armed": True}


# Test-only: force the next N /result calls to return success=false.
@app.post("/test/delay_next_results/{count}")
async def armDelayResults(count: int = Path(...)):
    DELAY_RESULTS_COUNT["remaining"] = count
    return {"remaining": count}


# Test-only: force the next GET /result call to return the given HTTP status.
# Consumed after one call.
@app.post("/test/force_next_result_status/{code}")
async def armForceResultStatus(code: int = Path(...)):
    FORCE_NEXT_RESULT_STATUS["code"] = code
    return {"armed_status": code}


# Test-only: reset all test-hook globals so tests are order-independent.
@app.post("/test/reset")
async def resetTestState():
    FAIL_NEXT_JOB["enabled"] = False
    DELAY_RESULTS_COUNT["remaining"] = 0
    FORCE_NEXT_RESULT_STATUS["code"] = None
    return {"reset": True}


# v2 API: GET /jobs/{job_qrn}
@app.get("/jobs/{job_id}")
async def getJob(
        job_id: str = Path(...),
        x_api_key: Optional[str] = Header(None, alias="X-API-KEY"),
):
    """Retrieve the status of a quantum job (v2 API)."""
    if x_api_key is None:
        raise HTTPException(status_code=401, detail="API key is required")

    job_data = poll_job_status(job_id)

    # v2 response: wrapped in success/data envelope
    return {"success": True, "data": job_data}


# v2 API: GET /jobs/{job_qrn}/program
@app.get("/jobs/{job_id}/program")
async def getJobProgram(
        job_id: str = Path(...),
        x_api_key: Optional[str] = Header(None, alias="X-API-KEY"),
):
    """Retrieve the program of a quantum job (v2 API)."""
    if x_api_key is None:
        raise HTTPException(status_code=401, detail="API key is required")

    if job_id not in JOBS_MOCK_DB:
        raise HTTPException(status_code=404, detail="Job not found")

    job_data = JOBS_MOCK_DB[job_id]

    # Return the stored program in v2 format: { success, data: { format, data } }
    return {
        "success": True,
        "data": {
            "format": job_data.get("program", {}).get("format", "qasm2"),
            "data": job_data.get("program", {}).get("data", ""),
        },
    }


# v2 API: GET /jobs/{job_qrn}/result
@app.get("/jobs/{job_id}/result")
async def getJobResult(
        job_id: str = Path(...),
        x_api_key: Optional[str] = Header(None, alias="X-API-KEY"),
):
    """Retrieve the results of a quantum job (v2 API)."""
    # Test hook: if armed, raise the requested status. Checked first so tests
    # can force 401/403 even when a valid api key is present.
    if FORCE_NEXT_RESULT_STATUS["code"] is not None:
        forced = FORCE_NEXT_RESULT_STATUS["code"]
        FORCE_NEXT_RESULT_STATUS["code"] = None
        raise HTTPException(status_code=forced,
                            detail=f"Forced HTTP {forced} for test")

    if x_api_key is None:
        raise HTTPException(status_code=401, detail="API key is required")

    if job_id not in JOBS_MOCK_DB:
        raise HTTPException(status_code=404, detail="Job not found")

    if JOBS_MOCK_DB[job_id]["status"] in {"FAILED", "CANCELLED"}:
        raise HTTPException(
            status_code=409,
            detail="Results unavailable. Job failed or was cancelled.")

    if JOBS_MOCK_DB[job_id]["status"] != "COMPLETED":
        # v2: use success=false instead of "error" field
        return {
            "success": False,
            "data": {
                "status": JOBS_MOCK_DB[job_id]["status"]
            },
        }

    if job_id not in JOBS_MOCK_RESULTS:
        raise HTTPException(status_code=500, detail="Job results not found")

    # Test hook: return "not yet available" for the next N /result calls if
    # the delay counter is armed. Decrements on each call.
    if DELAY_RESULTS_COUNT["remaining"] > 0:
        DELAY_RESULTS_COUNT["remaining"] -= 1
        return {
            "success": False,
            "data": {
                "status":
                    "COMPLETED",
                "message":
                    "Failed to retrieve job results. Please wait, and try again.",
            },
        }

    counts = JOBS_MOCK_RESULTS[job_id]

    # v2 response: measurementCounts nested under data.resultData
    return {
        "success": True,
        "data": {
            "resultData": {
                "measurementCounts": counts
            },
            "status": "COMPLETED",
            "cost": 0,
            "timeStamps": {},
        },
    }


def startServer(port):
    """Start the REST server."""
    uvicorn.run(app, port=port, host="0.0.0.0", log_level="info")


if __name__ == "__main__":
    startServer(62454)
