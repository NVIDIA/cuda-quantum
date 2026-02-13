# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq
from fastapi import FastAPI, HTTPException, Header, Request
from typing import Union
import uvicorn, uuid, base64, ctypes, sys
from pydantic import BaseModel
from llvmlite import binding as llvm
from cudaq.mlir.passmanager import PassManager
from cudaq.mlir.ir import Module
from cudaq.kernel.utils import getMLIRContext
from cudaq.mlir.dialects import func
from cudaq.mlir._mlir_libs._quakeDialects import cudaq_runtime

# Define the REST Server App
app = FastAPI()

llvm.initialize()
llvm.initialize_native_target()
llvm.initialize_native_asmprinter()
target = llvm.Target.from_default_triple()
targetMachine = target.create_target_machine()
backing_mod = llvm.parse_assembly("")
engine = llvm.create_mcjit_compiler(backing_mod, targetMachine)

# Keep track of Job Ids to their Names
createdJobs = {}


@app.post("/job")
async def postJob(request: Request):
    global createdJobs
    payload = await request.json()
    # Decode base64
    decoded_payload = base64.b64decode(payload["ir"]).decode('utf-8')

    # Remove "cudaq-autogen" attribute from all operations in the module
    # Note: This is needed as we reuse CUDA-Q pass (which has customization for this attribute for AOT compilation) for lowering here.
    # Real backend should have its own pass, which doesn't care about this attribute.
    decoded_payload = decoded_payload.replace('"cudaq-autogen", ', '')
    decoded_payload = decoded_payload.replace(', "cudaq-autogen"', '')
    decoded_payload = decoded_payload.replace('"cudaq-autogen"', '')

    ctx = getMLIRContext()
    recovered_mod = Module.parse(decoded_payload, context=ctx)
    pm = PassManager.parse(
        "builtin.module(canonicalize,distributed-device-call,cse)", context=ctx)
    try:
        pm.run(recovered_mod)
    except:
        raise RuntimeError(
            f"Failed to run pass manager on the recovered module.")

    entry_func_name = ""
    for op in recovered_mod.body.operations:
        if isinstance(op, func.FuncOp):
            for attr in op.attributes:
                if attr.name == "cudaq-entrypoint":
                    entry_func_name = op.name.value
                    break
    # Lower the module to LLVM IR
    qir_code = cudaq_runtime._lower_to_qir(recovered_mod)
    m = llvm.module.parse_assembly(qir_code)
    m.verify()

    # Job ID
    newId = str(uuid.uuid4())

    all_funcs = m.functions
    for f in all_funcs:
        if f.is_declaration:
            # Look up external functions: get the in-process address me.
            func_addr = llvm.address_of_symbol(f.name)
            if func_addr is None:
                createdJobs[
                    newId] = f"FAILURE: Function {f.name} not found in JIT engine."
                return ({"id": newId}, 201)

    # JIT Compile and get Function Pointer
    engine.add_module(m)
    engine.finalize_object()
    engine.run_static_constructors()
    funcPtr = engine.get_function_address(entry_func_name)
    kernel = ctypes.CFUNCTYPE(None)(funcPtr)
    # Clear any leftover log from previous jobs
    cudaq.testing.getAndClearOutputLog()
    qir_log = f"HEADER\tschema_id\tlabeled\nHEADER\tschema_version\t1.0\nSTART\nMETADATA\tentry_point\nMETADATA\tqir_profiles\tadaptive_profile\n"

    shots = payload["shots"]
    for i in range(shots):
        kernel()
        shot_log = cudaq.testing.getAndClearOutputLog()
        if i > 0:
            qir_log += "START\n"
        qir_log += shot_log
        qir_log += "END\t0\n"

    engine.remove_module(m)
    createdJobs[newId] = qir_log
    # Job "created", return the id
    return ({"id": newId}, 201)


@app.get("/job/{jobId}")
async def getJob(jobId: str):
    if jobId not in createdJobs:
        raise HTTPException(status_code=404, detail="Job ID not found")

    job_output = createdJobs[jobId]
    if job_output.startswith("FAILURE:"):
        res = ({"status": "error", "message": job_output}, 200)
    else:
        res = ({"status": "done", "qir_output": job_output}, 201)
    return res


def startServer(port):
    print("Server Started")
    uvicorn.run(app, port=port, host='0.0.0.0', log_level="debug")


if __name__ == '__main__':
    args = sys.argv[1:]

    # Load the device library if provided as an argument
    if len(args) == 2 and args[0] == '-device-lib':
        print(f"Loading device library from {args[1]}")
        llvm.load_library_permanently(args[1])

    print("Server Starting")
    startServer(62451)
