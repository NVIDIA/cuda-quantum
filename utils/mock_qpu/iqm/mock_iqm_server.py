# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
import asyncio
import uuid
from typing import Optional
import math
from cmath import exp

# Use IQM Client Tools to verify data structures
import iqm.iqm_client as iqm_client
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import numpy as np

# Testing constants
good_access_token = "Bearer good_access_token"
server_qpu_architecture = "Apollo"
operations = []  # TBA
qubits = []  # TBA
qubit_connectivity = []  # TBA

# Define the REST Server App
app = FastAPI()


class Counts(BaseModel):
    """State histogram"""

    measurement_keys: list[str]
    counts: dict[str, int]


# Keep job artifacts
class Job(BaseModel):
    """Job information"""

    id: str
    status: iqm_client.Status
    result: Optional[iqm_client.RunResult] = None
    counts_batch: Optional[list[Counts]] = None
    metadata: iqm_client.Metadata


# New job created response
class PostJobsResponse(BaseModel):
    """POST /jobs response"""

    id: str


# Jobs storage
createdJobs: dict[str, Job] = {}


def _contract_einsum(A: np.ndarray, U: np.ndarray, indices: list[int],
                     a_dims: list[int], arity):
    """Unitary operator A acting on the given subsystems of the register,
    multiplied by the full-register propagator U."""
    A = A.reshape(2 * a_dims)
    u_inds = np.arange(2 * arity)

    # some u indexes are contracted and replaced with new indices
    new_inds = np.arange(len(a_dims)) + len(u_inds)
    a_inds = list(new_inds) + indices

    # output indexes are same as input indexes, but with the contracted ones replaced with the new ones
    out_inds = u_inds.copy()
    out_inds[indices] = new_inds

    return np.einsum(A, a_inds, U, u_inds, out_inds)


def _generate_measurement_strings(n, bs=""):
    if n - 1:
        yield from _generate_measurement_strings(n - 1, bs + "0")
        yield from _generate_measurement_strings(n - 1, bs + "1")
    else:
        yield bs + "0"
        yield bs + "1"


def _make_phased_rx_unitary_matrix(theta: float, phi: float) -> np.ndarray:
    """Return the unitary matrix for a phased RX gate."""
    cos = math.cos(theta / 2)
    sin = math.sin(theta / 2)
    exp_m = exp(-1j * phi)
    exp_p = exp(1j * phi)
    r_gate = np.array([[cos, -1j * exp_m * sin], [-1j * exp_p * sin, cos]])
    return r_gate


def _make_cz_unitary_matrix() -> np.ndarray:
    """Return the unitary matrix for a CZ gate."""
    CZ = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]])
    return CZ


def _extract_qubit_position_from_qubit_name(qubit_names: str) -> int:
    """Extract the qubit position from the qubit name."""
    return int(qubit_names[2:]) - 1


def _partial_trace(N, rho, keep):
    """Calculate the partial trace of a density matrix"""
    trace_out = sorted(set(range(N)) - set(keep), reverse=True)

    if len(trace_out) == 0:
        return rho.reshape(
            2**N, 2**N)  # No tracing needed, return the reshaped matrix

    # Reshape into tensor with shape (2,2,...,2,2,...,2), 2N times
    rho = rho.reshape([2] * 2 * N)

    # Trace over the unwanted qubits
    for q in trace_out:
        rho = np.trace(rho, axis1=q, axis2=q + N)
        N -= 1  # Adjust N as one qubit is traced out

    return rho


def _validate_measurements(job: Job, circuit: iqm_client.Circuit) -> bool:
    """Check that the circuit contains measurements"""
    measurements = [
        instruction for instruction in circuit.instructions
        if instruction.name == "measurement"
    ]
    if len(measurements) == 0:
        job.status = iqm_client.Status.FAILED
        job.result = iqm_client.RunResult(
            status=job.status,
            metadata=job.metadata,
            message="Circuit contains no measurements",
        )
        createdJobs[job.id] = job
        return False
    return True


def _validate_connectivity(job: Job, circuit: iqm_client.Circuit) -> bool:
    """C""check connectivity partially matches Apollo"""
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
            message=
            "Some circuits in the batch have gates between uncoupled qubits:",
        )
        createdJobs[job.id] = job
        return False
    return True


def _gather_circuit_information(
    instructions: list[iqm_client.Instruction],) -> tuple[set[int], int]:
    """Gather qubits from the circuit"""
    measurement_qubits: set[int] = set()
    all_qubits: set[int] = set()
    for instruction in instructions:
        all_qubits.update(
            _extract_qubit_position_from_qubit_name(qb)
            for qb in list(instruction.qubits))
        if instruction.name == "measurement":
            measurement_qubits.update(
                _extract_qubit_position_from_qubit_name(qb)
                for qb in list(instruction.qubits))
    return measurement_qubits, len(all_qubits)


def _simulate_circuit(instructions: list[iqm_client.Instruction],
                      shots: int) -> dict[str, int]:
    """Simulate the circuit"""
    # extract qubits information from measurements
    measurement_qubits_positions, number_of_qubits = _gather_circuit_information(
        instructions)

    # calculate circuit operator and measure qubits
    dims = [2] * number_of_qubits
    D = np.prod(dims)
    operator: np.ndarray = np.eye(int(D), dtype=complex)
    operator = operator.reshape(2 * dims)

    for instruction in instructions:
        if instruction.name == "phased_rx":
            qubit_position = _extract_qubit_position_from_qubit_name(
                instruction.qubits[0])
            r_gate = _make_phased_rx_unitary_matrix(
                float(instruction.args["angle_t"]) * (2.0 * np.pi),
                float(instruction.args["phase_t"]) * (2.0 * np.pi),
            )

            # arity here is `number_of_qubits` because `operator` is an operation over all the qubits
            operator = _contract_einsum(r_gate, operator, [qubit_position],
                                        [2] * 1, number_of_qubits)
        elif instruction.name == "cz":
            control_qubit_position = _extract_qubit_position_from_qubit_name(
                instruction.qubits[0])
            target_qubit_position = _extract_qubit_position_from_qubit_name(
                instruction.qubits[1])
            cz_gate = _make_cz_unitary_matrix()

            # arity here is `number_of_qubits` because `operator` is an operation over all the qubits
            operator = _contract_einsum(
                cz_gate,
                operator,
                [control_qubit_position, target_qubit_position],
                [2] * 2,
                number_of_qubits,
            )
        else:
            continue

    operator = operator.reshape((D, D))

    # apply the constructed operator to the initial state
    initial_state = np.array([0] * 2**number_of_qubits, dtype=complex)
    initial_state[0] = 1
    final_state = np.matmul(operator, initial_state)

    # density matrix
    density_matrix = np.outer(final_state, np.conj(final_state))

    # make partial density matrix for the measured subset of qubits
    partial_trace = _partial_trace(number_of_qubits, density_matrix,
                                   measurement_qubits_positions)
    probabilities = np.diag(partial_trace)
    return {
        ms: int(prob * shots) for ms, prob in zip(
            _generate_measurement_strings(len(measurement_qubits_positions)),
            probabilities,
        )
    }


async def compile_and_submit_job(job: Job):
    """Analyze measurements and construct corresponding counts"""
    request = job.metadata.request
    circuits = request.circuits

    job.counts_batch = []
    for circuit in circuits:
        if not _validate_measurements(job, circuit):
            return

        if not _validate_connectivity(job, circuit):
            return

        # Simulate the circuit
        counts = _simulate_circuit(circuit.instructions, request.shots)

        job.counts_batch.append(
            Counts(counts=counts, measurement_keys=[circuit.name]))

    job.status = iqm_client.Status.READY
    job.result = iqm_client.RunResult(status=job.status, metadata=job.metadata)
    createdJobs[job.id] = job


@app.get("/quantum-architecture")
async def get_quantum_architecture(
        request: Request) -> iqm_client.QuantumArchitecture:
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
        ))


@app.post("/jobs")
async def post_jobs(job_request: iqm_client.RunRequest,
                    request: Request) -> PostJobsResponse:
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

    # start compilation and execution
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


@app.get("/jobs/{job_id}/counts")
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
        "status":
            job.status,
        "message":
            job.result.message if job.result and job.result.message else None,
        "counts_batch":
            job.counts_batch,
    }

    return results


def startServer(port):
    uvicorn.run(app, port=port, host="0.0.0.0", log_level="debug")


if __name__ == "__main__":
    startServer(9100)
