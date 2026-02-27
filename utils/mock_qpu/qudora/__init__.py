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
import uvicorn, uuid, base64, ctypes
from pydantic import BaseModel
from llvmlite import binding as llvm
import json
import base64
import zlib

# Define the REST Server App
app = FastAPI()


class InputJob(BaseModel):
    name: str
    language: str
    shots: list[int]
    target: str
    input_data: list[str]
    backend_settings: str | None


# Jobs look like the following type
class JobStatus(BaseModel):
    job_id: int
    name: str
    result: list[str]
    shots: list[int]
    status: str
    target: str


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
                                 attribute_names: list[str]) -> int | None:
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
    def get_num_required_qubits(cls, function) -> int | None:
        """Extract required number of qubits from function attributes"""
        return cls._extract_attribute_value(
            function, ["required_num_qubits", "requiredQubits"])

    @classmethod
    def get_num_required_results(cls, function) -> int | None:
        """Extract required number of results from function attributes"""
        return cls._extract_attribute_value(
            function, ["required_num_results", "requiredResults"])


# Keep track of Job Ids to their Names
createdJobs = {}

# Count how many times the client has requested the Job
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


def simulate_qir(program: str, shots: int) -> str:
    decoded = base64.b64decode(program)
    m = llvm.module.parse_bitcode(decoded)
    mstr = str(m)
    assert ('entry_point' in mstr)

    analyzer = KernelAnalyzer()

    # Get the function, number of qubits, and kernel name
    function = analyzer.get_kernel_function(m)
    if function == None:
        raise Exception("Could not find kernel function")
    kernelFunctionName = function.name
    numQubitsRequired = analyzer.get_num_required_qubits(function) or 0
    numResultsRequired = analyzer.get_num_required_results(function) or 0

    print("Kernel name = ", kernelFunctionName)
    print("Requires {} qubits".format(numQubitsRequired))

    # JIT Compile and get Function Pointer
    engine.add_module(m)
    engine.finalize_object()
    engine.run_static_constructors()
    funcPtr = engine.get_function_address(kernelFunctionName)
    kernel = ctypes.CFUNCTYPE(None)(funcPtr)

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

    engine.remove_module(m)

    compressed_log = base64.b64encode(zlib.compress(qir_log.encode())).decode()

    return compressed_log


# Here we expose a way to post jobs,
# Must have a Access Token, Job Program must be Adaptive Profile
# with entry_point tag
@app.post("/jobs")
async def postJob(job: InputJob,
                  token: Union[str, None] = Header(alias="Authorization",
                                                   default=None)):
    global createdJobs, numQubitsRequired

    if token == None:
        raise HTTPException(401, detail="Credentials not provided")

    print('Posting job with shots = ', job.shots)
    newId = str(uuid.uuid4())

    results = [
        simulate_qir(program, shots)
        for program, shots in zip(job.input_data, job.shots)
    ]

    createdJobs[newId] = results
    print("Adding job results to id", newId)

    # Job "created", return the id
    return newId


# Retrieve the job, simulate having to wait by counting to 3
# until we return the job results
@app.get("/jobs")
async def getJob(job_id: str, include_results: bool):
    global countJobGetRequests, createdJobs, numQubitsRequired

    assert include_results, "include_results=False not implemented."

    # Simulate asynchronous execution
    if countJobGetRequests < 3:
        countJobGetRequests += 1
        return [{"status": "Running"}]

    countJobGetRequests = 0

    job_id = json.loads(job_id)

    print("Requesting job status for id", job_id)

    res = [{"status": "Completed", "qir_result": createdJobs[job_id]}]
    return res


def startServer(port):
    uvicorn.run(app, port=port, host='0.0.0.0', log_level="info")
