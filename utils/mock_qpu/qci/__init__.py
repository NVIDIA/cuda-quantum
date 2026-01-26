# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates and Contributors.  #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import base64
import ctypes
import uuid
from typing import Any, Optional, Union

import cudaq
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import PlainTextResponse
from llvmlite import binding as llvm
from pydantic import BaseModel
from .. import get_backend_port

# Define the REST Server App
app = FastAPI()

# In-memory storage
createdJobs = {}
jobResults = {}
jobStatuses = {}
jobCountRequests = {}

# Count how many times the client has requested the Job
countJobGetRequests = 0
# Save how many qubits were needed for each test (emulates real backend)
numQubitsRequired = 0


# Mock QCI API Models
class JobRequest(BaseModel):
    code: str
    machine: str
    mappingReorderIdx: Optional[list[int]] = None
    name: str
    outputNames: Optional[Any] = None
    userData: Optional[Any] = None
    options: dict[str, Any] = {}


llvm.initialize()
llvm.initialize_native_target()
llvm.initialize_native_asmprinter()
target = llvm.Target.from_default_triple()
targetMachine = target.create_target_machine()
backing_mod = llvm.parse_assembly("")
engine = llvm.create_mcjit_compiler(backing_mod, targetMachine)


class KernelAnalyzer:
    """Analyzes LLVM modules to extract kernel information"""

    @staticmethod
    def get_kernel_function(module):
        """Find the main kernel function in the module"""
        for func in module.functions:
            if not func.is_declaration:
                return func
        return None

    @staticmethod
    def _extract_attribute_value(function,
                                 attribute_names: list[str]) -> Optional[int]:
        """Extract integer value from function attributes"""
        for attr in function.attributes:
            attr_str = str(attr)
            for attr_name in attribute_names:
                if attr_name in attr_str:
                    try:
                        value = attr_str.split(f'{attr_name}"=')[-1].split(
                            " ")[0]
                        return int(value.replace('"', '').replace("'", ""))
                    except (IndexError, ValueError):
                        continue
        return None

    @classmethod
    def get_num_required_qubits(cls, function) -> Optional[int]:
        """Extract required number of qubits from function attributes"""
        return cls._extract_attribute_value(
            function, ["required_num_qubits", "requiredQubits"])

    @classmethod
    def get_num_required_results(cls, function) -> Optional[int]:
        """Extract required number of results from function attributes"""
        return cls._extract_attribute_value(
            function, ["required_num_results", "requiredResults"])


# Here we expose a way to post jobs,
# Must have a Access Token, Job Program must be Adaptive Profile
# with entry_point tag
@app.post("/cudaq/v1/jobs")
async def postJob(job: JobRequest,
                  token: Union[str, None] = Header(alias="Authorization",
                                                   default=None)):
    global createdJobs, shots, numQubitsRequired

    if token == None:
        raise HTTPException(status_code(401), detail="Credentials not provided")

    n_shots = job.options.get("aqusim", {}).get("shots", 1000)
    print('Posting job with shots = ', n_shots)
    jobId = str(uuid.uuid4())
    jobName = job.name
    shots = n_shots
    program = job.code
    decoded = base64.b64decode(program)
    m = llvm.module.parse_bitcode(decoded)
    mstr = str(m)
    assert ('entry_point' in mstr)

    analyzer = KernelAnalyzer()

    # Get the function, number of qubits, and kernel name
    function = analyzer.get_kernel_function(m)
    if function == None:
        raise Exception("Could not find kernel function")
    numQubitsRequired = analyzer.get_num_required_qubits(function) or 0
    numResultsRequired = analyzer.get_num_required_results(function) or 0
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
    # NOTE: This uses QIR v1.0
    qir_log = f"HEADER\tschema_id\tlabeled\nHEADER\tschema_version\t1.0\nSTART\nMETADATA\tentry_point\nMETADATA\tqir_profiles\tadaptive_profile\nMETADATA\trequired_num_qubits\t{numQubitsRequired}\nMETADATA\trequired_num_results\t{numResultsRequired}\n"
    for i in range(shots):
        cudaq.testing.toggleDynamicQubitManagement()
        qubits, context = cudaq.testing.initialize(numQubitsRequired, 1, "run")
        kernel()
        _ = cudaq.testing.finalize(qubits, context)

        shot_log = cudaq.testing.getAndClearOutputLog()
        if i > 0:
            qir_log += "START\n"
        qir_log += shot_log
        qir_log += "END\t0\n"

    createdJobs[jobId] = (jobName, qir_log)

    engine.remove_module(m)

    # Job "created", return the id
    return {"id": jobId, "jobs": {"status": "running"}}


@app.get("/cudaq/v1/jobs/{job_id}")
async def get_job_status(job_id: str):
    global createdJobs, createdResults
    port = get_backend_port("qci")

    if job_id not in createdJobs:
        raise HTTPException(status_code=404, detail="Job not found")

    # Simulate job processing time
    jobCountRequests[job_id] = jobCountRequests.get(job_id, 0) + 1

    # After a few requests, mark the job as completed
    if jobCountRequests[job_id] >= 3:
        jobStatuses[job_id] = "completed"

        # Return completed job with result URL
        return {
            "id": job_id,
            "status": "completed",
            "exited": True,
            "outputUrl": f"http://localhost:{port}/cudaq/v1/results/{job_id}"
        }

    # Job still running
    return {"id": job_id, "status": "running", "exited": False}


@app.get("/cudaq/v1/results/{job_id}")
async def get_job_results(job_id: str):
    global createdJobs, jobResults, numQubitsRequired
    if job_id not in createdJobs:
        raise HTTPException(status_code=404, detail="Job not found")

    if job_id not in jobResults:
        # Prepare and store results
        _, qir_log = createdJobs[job_id]
        jobResults[job_id] = qir_log

    return PlainTextResponse(content=jobResults[job_id],
                             media_type="text/tab-separated-values")
