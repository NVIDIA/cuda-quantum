# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
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
from os import environ

# Use IQM Client Tools to verify data structures
import iqm.iqm_client as iqm_client
from fastapi import FastAPI, HTTPException, Request
from contextlib import asynccontextmanager
from pydantic import BaseModel
import numpy as np

# Testing constants
good_access_token = "Bearer good_access_token"

bad_qubits_prx = []
"""
To simulate a QPU with an imperfect calibration this list can be used to
deliberately exclude `prx` gates from the dynamic quantum architecture.
By default 2 qubits are excluded which the integration needs to skip over
in order for tests to succeed. This list can be set at startup by assigning
a list to the environment variable IQM_MOCK_BAD_PRX_GATES. It can also be
extended dynamically at runtime using HTTP requests to endpoints defined below.
"""

bad_cz_gates = []
"""
Similar to `bad_qubits_prx` this lists all the CZ gates which should be removed
from the dynamic quantum architecture to simulate an imperfect calibration.
This list can be set at startup by assigning a list to the environment variable
IQM_MOCK_BAD_CZ_GATES. It can also be extended dynamically at runtime using
HTTP requests to endpoints defined below.
"""

qubits = []

qubit_connectivity = []

computational_resonators = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Once on server start get configuration from the environment.
       Setting the environment variable `IQM_MOCK_BAD_PRX_GATES` with a comma
       separated list of qubit names will remove these qubits from the list
       of qubits with PRX functionality in the dynamic quantum architecture.
       Example: `export IQM_MOCK_BAD_PRX_GATES="QB2,QB5"`

       Setting the environment variable `IQM_MOCK_BAD_CZ_GATES` with a comma
       separated list of qubit name pairs separated by hyphen will remove
       matching cz-gates from the list of gates with CZ functionality in the
       dynamic quantum architecture.
       Example: `export IQM_MOCK_BAD_CZ_GATES="QB1-QB2,QB5-QB6,QB19-QB20"`
    """

    # default quantum architecture on startup
    _generate_quantum_architecture("crystal-20")

    if ("IQM_MOCK_BAD_PRX_GATES" in environ):
        # Allow setting the list of bad PRX gates via environment variable.
        bad_qubits_prx.clear()
        gate_list = environ["IQM_MOCK_BAD_PRX_GATES"]
        _parse_bad_prx_gate_list(gate_list)

    for qb in bad_qubits_prx:
        print(f"Disabled PRX-gate on: {qb}")

    if ("IQM_MOCK_BAD_CZ_GATES" in environ):
        # Likewise a list of bad CZ gates can be given.
        bad_cz_gates.clear()
        gate_list = environ["IQM_MOCK_BAD_CZ_GATES"]
        _parse_bad_cz_gate_list(gate_list)

    _process_bad_cz_gate_list()

    yield


# Define the REST Server App
app = FastAPI(title="IQM Mock QPU Server", version="V1", lifespan=lifespan)


def _parse_bad_prx_gate_list(gate_list: str):
    """Parse the given list and update the global var bad_qubits_prx."""
    for qb in gate_list.split(","):
        qb = qb.strip()
        if qb in qubits and qb not in bad_qubits_prx:
            bad_qubits_prx.append(qb)


def _parse_bad_cz_gate_list(gate_list: str):
    """Parse the given list and update the global var bad_cz_gates."""
    for gate in gate_list.split(","):
        cz_qubits: list[str] = gate.split("-")
        if len(cz_qubits) == 2 and cz_qubits not in bad_cz_gates:
            bad_cz_gates.append(cz_qubits)


def _process_bad_cz_gate_list():
    """Process the global var bad_cz_gates and remove the listed cz-gates from
    the qubit connectivity list."""
    for gate in bad_cz_gates:
        if gate in qubit_connectivity:
            qubit_connectivity.remove(gate)
            print(f"Disabled CZ-gate between: {gate[0]}-{gate[1]}")
        else:
            reverse_gate: list[str] = gate.copy()
            reverse_gate.reverse()
            if reverse_gate in qubit_connectivity:
                qubit_connectivity.remove(reverse_gate)
                print(f"Disabled CZ-gate between: "
                      f"{reverse_gate[0]}-{reverse_gate[1]}")


def _generate_quantum_architecture(qpu: str) -> bool:
    """Generate the quantum architecture description of the specified QPU.
    This populates the global variables 'qubits' and 'qubit_connectivity'."""
    architectures = {
        "crystal-5": [1, 0, 3, 1, 1, -1],
        "crystal-20": [2, 0, 5, 1, 5, 0, 5, 0, 3, -1],
        "crystal-54": [2, 0, 5, 2, 7, 1, 8, 1, 9, 0, 8, -1, 7, 0, 5, -1, 3, -1]
    }
    if qpu not in architectures:
        return False

    qubits.clear()
    qubit_connectivity.clear()
    layout = architectures[qpu]

    # generate the list of two qubit gates
    row_start = 1
    last_row_start = 0
    last_row_length = 0

    for r in range(0, len(layout), 2):
        row_length = layout[r]
        row_offset = layout[r + 1]

        # horizontal
        for qb in range(0, row_length - 1):
            qb1 = row_start + qb
            qb2 = qb1 + 1
            qubit_connectivity.append([f"QB{qb1}", f"QB{qb2}"])

        # vertical
        if last_row_start != 0:
            for qb in range(0, min(last_row_length, row_length)):
                if row_offset >= 0:
                    qb1 = last_row_start + qb
                    qb2 = row_start + row_offset + qb
                else:
                    qb1 = last_row_start - row_offset + qb
                    qb2 = row_start + qb
                qubit_connectivity.append([f"QB{qb1}", f"QB{qb2}"])

        last_row_length = row_length
        last_row_start = row_start
        row_start += row_length

    # generate the list of qubits
    qubit_cnt = row_start - 1
    qubits.extend(f"QB{qb + 1}" for qb in range(qubit_cnt))

    return True


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


def _extract_qubit_position_from_qubit_name(qubit_name: str) -> int:
    """Extract the qubit position from the qubit name."""
    return int(qubit_name[2:]) - 1


def _partial_trace(N, rho, keep):
    """Calculate the partial trace of a density matrix"""
    trace_out = sorted(set(range(N)) - set(keep), reverse=True)

    if len(trace_out):

        # Reshape into tensor with shape (2,2,...,2,2,...,2), 2N times
        rho = rho.reshape([2] * 2 * N)

        # Trace over the unwanted qubits
        for q in trace_out:
            rho = np.trace(rho, axis1=q, axis2=q + N)
            N -= 1  # Adjust N as one qubit is traced out

    # Return the reshaped matrix
    return rho.reshape(2**N, 2**N)


def _validate_measurements(job: Job, circuit: iqm_client.Circuit) -> bool:
    """Check that the circuit contains measurements"""
    measurements = [
        instruction for instruction in circuit.instructions
        if instruction.name == "measure"
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
    """Check connectivity matches the qpu-architecture"""
    request = job.metadata.request
    qubit_mapping: Optional[dict[str, str]] = None

    if (request.qubit_mapping is not None) and (request.qubit_mapping):
        qubit_mapping = {}
        for sqm in request.qubit_mapping:
            qubit_mapping[sqm.logical_name] = sqm.physical_name

    for instruction in circuit.instructions:
        if len(instruction.qubits) == 2:
            qubit_pair = list(instruction.qubits)
            if qubit_mapping is not None:
                qubit_pair[0] = qubit_mapping[qubit_pair[0]]
                qubit_pair[1] = qubit_mapping[qubit_pair[1]]
            reverse_qubit_pair = qubit_pair.copy()
            reverse_qubit_pair.reverse()

            if qubit_pair not in qubit_connectivity \
               and reverse_qubit_pair not in qubit_connectivity:
                # qubit combination not found in architecture -> abort
                job.status = iqm_client.Status.FAILED
                job.result = iqm_client.RunResult(
                    status=job.status,
                    metadata=job.metadata,
                    message="Some circuits in the batch have gates between" +
                    " uncoupled qubits: " + "-".join(qubit_pair),
                )
                createdJobs[job.id] = job
                return False
    return True


def _gather_circuit_information(
    instructions: list[iqm_client.Instruction],
) -> tuple[set[int], dict[int, str], int]:
    """Gather qubits from the circuit"""
    measurement_qubits: set[int] = set()
    measurement_keys: dict[int, str] = dict()
    all_qubits: set[int] = set()
    for instruction in instructions:
        all_qubits.update(
            _extract_qubit_position_from_qubit_name(qb)
            for qb in list(instruction.qubits))
        if instruction.name == "measure":
            measurement_qubits.update(
                _extract_qubit_position_from_qubit_name(qb)
                for qb in list(instruction.qubits))
            measurement_keys[_extract_qubit_position_from_qubit_name(
                instruction.qubits[0])] = instruction.args["key"]
    return measurement_qubits, measurement_keys, len(all_qubits)


def _simulate_circuit(instructions: list[iqm_client.Instruction],
                      shots: int) -> tuple[dict[str, int], dict[int, str]]:
    """Simulate the circuit"""
    # extract qubits information from measurements
    measurement_qubits_positions, measurement_keys, number_of_qubits = \
        _gather_circuit_information(instructions)

    # calculate circuit operator and measure qubits
    dims = [2] * number_of_qubits
    D = np.prod(dims)
    operator: np.ndarray = np.eye(int(D), dtype=complex)
    operator = operator.reshape(2 * dims)

    for instruction in instructions:
        if instruction.name == "prx":
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
        ms: int(np.round(np.real(prob * shots))) for ms, prob in zip(
            _generate_measurement_strings(len(measurement_qubits_positions)),
            probabilities,
        )  # if np.real(prob * shots) >= 1  # to suppress < 1 shots occurrences
    }, measurement_keys


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
        counts, mkeys = _simulate_circuit(circuit.instructions, request.shots)

        # {"counts":{"0":504,"1":496},"measurement_keys":["m_QB1"]}
        job.counts_batch.append(
            Counts(counts=counts,
                   measurement_keys=[mkeys[key] for key in sorted(mkeys)]))

    job.status = iqm_client.Status.READY
    job.result = iqm_client.RunResult(status=job.status, metadata=job.metadata)
    createdJobs[job.id] = job


@app.get("/api/v1/calibration-sets/{qc}/default/dynamic-quantum-architecture")
async def get_dynamic_quantum_architecture(
        request: Request) -> iqm_client.DynamicQuantumArchitecture:
    """Get the dynamic quantum architecture"""

    access_token = request.headers.get("Authorization")
    if access_token != good_access_token:
        raise HTTPException(401)

    return iqm_client.DynamicQuantumArchitecture(
        calibration_set_id=str(uuid.uuid4()),
        qubits=qubits,
        computational_resonators=computational_resonators,
        gates={
            "cz":
                iqm_client.GateInfo(
                    implementations={
                        "crf_crf":
                            iqm_client.GateImplementationInfo(loci=tuple(
                                tuple(pair) for pair in qubit_connectivity)),
                    },
                    default_implementation="crf_crf",
                    override_default_implementation={},
                ),
            "measure":
                iqm_client.GateInfo(
                    implementations={
                        "constant":
                            iqm_client.GateImplementationInfo(loci=tuple(
                                (qubit,) for qubit in qubits))
                    },
                    default_implementation="constant",
                    override_default_implementation={},
                ),
            "prx":
                iqm_client.GateInfo(
                    implementations={
                        "drag_crf":
                            iqm_client.GateImplementationInfo(loci=tuple(
                                (qubit,)
                                for qubit in qubits
                                if qubit not in bad_qubits_prx))
                    },
                    default_implementation="drag_crf",
                    override_default_implementation={},
                ),
        })


@app.post("/api/v1/jobs/{qc}/circuit")
async def post_job(job_request: iqm_client.RunRequest,
                   request: Request) -> PostJobsResponse:
    """Register a new job and start execution"""

    access_token = request.headers.get("Authorization")
    if access_token != good_access_token:
        raise HTTPException(401)

    metadata = iqm_client.Metadata(request=job_request)
    new_job_id = str(uuid.uuid4())
    new_job = Job(
        id=new_job_id,
        status=iqm_client.Status.COMPILATION_STARTED,
        request=job_request,
        metadata=metadata,
    )
    createdJobs[new_job_id] = new_job

    # start compilation and execution
    asyncio.create_task(compile_and_submit_job(new_job))
    await asyncio.sleep(0.0)

    return PostJobsResponse(id=new_job_id)


@app.get("/api/v1/jobs/{job_id}")
async def get_job_status(job_id: str, request: Request):
    """Get the status of a job"""

    access_token = request.headers.get("Authorization")
    if access_token != good_access_token:
        raise HTTPException(401)

    if job_id not in createdJobs:
        raise HTTPException(404)

    job = createdJobs[job_id]

    results = {
        # Note: this is a subset of what a real server would return.
        "artifacts": [],
        "messages": [],
        "queue_position":
            1,
        "runtime_ms":
            None,
        "status":
            "completed"
            if job.status == iqm_client.Status.READY else job.status,
        "message":
            job.result.message if job.result and job.result.message else None,
        "counts_batch":
            job.counts_batch,
        "metadata":
            job.metadata,
    }

    if job.status == iqm_client.Status.FAILED:
        results["errors"] = list()
        results["errors"].append({
            "error_code": "unknown",
            "message": job.result.message,
            "source": "iqm-server"
        })

    return results


@app.get("/api/v1/jobs/{job_id}/payload")
async def get_job_payload(job_id: str, request: Request):
    """Get the payload of a job"""

    access_token = request.headers.get("Authorization")
    if access_token != good_access_token:
        raise HTTPException(401)

    if job_id not in createdJobs:
        raise HTTPException(404)

    job = createdJobs[job_id]
    return job.metadata.request


@app.get("/api/v1/jobs/{job_id}/artifacts/measurement_counts")
async def get_job_counts(job_id: str, request: Request):
    """Get the result of a job"""
    access_token = request.headers.get("Authorization")
    if access_token != good_access_token:
        raise HTTPException(401)

    if job_id not in createdJobs:
        raise HTTPException(404)

    job = createdJobs[job_id]

    return job.counts_batch


@app.get("/config/qa/qpu")
async def set_qa_qpu(qpu: str, request: Request):
    """Set the quantum architecture by selecting a QPU"""
    access_token = request.headers.get("Authorization")
    if access_token != good_access_token:
        raise HTTPException(401)

    status = _generate_quantum_architecture(qpu)
    if not status:
        raise HTTPException(404, "Requested QPU not found")

    bad_qubits_prx.clear()
    bad_cz_gates.clear()

    print(f"Using QPU architecture {qpu} now.")
    return {"message": "ok"}


@app.get("/config/qa/bad-prx-gates")
async def set_qa_bad_qubits_prx(loci_list: str, request: Request):
    """Set a list of qubits which cannot be used for PRX gates."""
    access_token = request.headers.get("Authorization")
    if access_token != good_access_token:
        raise HTTPException(401)

    _parse_bad_prx_gate_list(loci_list)
    for qb in bad_qubits_prx:
        print(f"Disabled PRX-gate on: {qb}")
    return {"message": "ok"}


@app.get("/config/qa/bad-cz-gates")
async def set_qa_bad_cz_gates(loci_list: str, request: Request):
    """Set a list of cz-gates which cannot be used."""
    access_token = request.headers.get("Authorization")
    if access_token != good_access_token:
        raise HTTPException(401)

    _parse_bad_cz_gate_list(loci_list)
    _process_bad_cz_gate_list()
    return {"message": "ok"}


def startServer(port):
    import uvicorn
    uvicorn.run(app, port=port, host='0.0.0.0', log_level="info")
