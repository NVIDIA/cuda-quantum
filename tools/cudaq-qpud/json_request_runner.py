# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Set `MPLCONFIGDIR` if running as nobody in order to prevent a warning message
# that is telling the truth about extended loading times.
import os
if 'nonexistent' in os.environ['HOME']:
    os.environ['MPLCONFIGDIR'] = os.getcwd()

import cudaq
import sys
import json
import subprocess
import importlib
from datetime import datetime
import re

# Pattern to detect ANSI escape color code in the error message
ANSI_PATTERN = re.compile(r'\x1B[@-_][0-?]*[ -/]*[@-~]')


def get_deserialized_dict(scoped_dict):
    deserialized_dict = {}

    # If the scoped_dict is one big JSON string, then load it into a
    # dictionary-like object.
    if isinstance(scoped_dict, str):
        scoped_dict = json.loads(scoped_dict)

    # Do two passes. Save the unpacking of cudaq.kernels for the second pass so
    # that they can see and utilize global variables unpacked in the first pass.
    for p in range(2):
        isFirstPass = (p == 0)
        for key, val in scoped_dict.items():
            isKernel = "/" in key and ".PyKernelDecorator" in key
            try:
                if "/" in key and ((isFirstPass and not isKernel) or
                                   (not isFirstPass is isKernel)):
                    key, val_type = key.split('/')
                    if val_type.startswith('cudaq.'):
                        module_name, type_name = val_type.rsplit('.', 1)
                        module = importlib.import_module(module_name)
                        type_class = getattr(module, type_name)
                        if isFirstPass:
                            result = type_class.from_json(json.dumps(val))
                        else:
                            result = type_class.from_json(
                                json.dumps(val), deserialized_dict)
                        deserialized_dict[key] = result
                    else:
                        raise Exception(f'Invalid val_type in key: {val_type}')
                elif isFirstPass:
                    deserialized_dict[key] = val
            except Exception as e:
                raise Exception(f"Error deserializing key '{key}': {e}")

    return deserialized_dict


if __name__ == "__main__":
    try:
        requestStart = int(datetime.now().timestamp() * 1000)

        # Expected command-line arguments:
        # `sys.argv[0] = json_request_runner.py`
        # `sys.argv[1] = <json file>`
        # `sys.argv[2] = --use-mpi=<0|1>`
        if '--use-mpi=1' in sys.argv:
            cudaq.mpi.initialize()

        # Read request
        if len(sys.argv) < 3:
            raise (Exception('Too few command-line arguments'))
        jsonFile = sys.argv[1]
        with open(jsonFile, 'rb') as fp:
            request = json.load(fp)

        serialized_ctx = request['serializedCodeExecutionContext']
        source_code = serialized_ctx['source_code']

        # Limit imports for the user code to a small subset of possible imports.
        imports_code = '\n'.join([
            'import cudaq', 'from cudaq import spin', 'import math',
            'import numpy', 'import numpy as np',
            'from typing import List, Tuple'
        ])

        # Be sure to do this before running any code from `serialized_ctx`
        globals_dict = get_deserialized_dict(serialized_ctx['scoped_var_dict'])

        # Determine which target to set
        sim2target = {
            'qpp': 'qpp-cpu',
            'custatevec_fp32': 'nvidia',
            'custatevec_fp64': 'nvidia-fp64',
            'tensornet': 'tensornet',
            'tensornet_mps': 'tensornet-mps',
            'dm': 'density-matrix-cpu',
            'nvidia_mgpu': 'nvidia-mgpu',
            'nvidia_mqpu': 'nvidia-mqpu',
            'nvidia_mqpu-fp64': 'nvidia-mqpu-fp64'
        }
        simulator_name = request['simulator']
        simulator_name = simulator_name.replace('-', '_')
        target_name = sim2target[simulator_name]

        # Validate the full source code
        full_source = f'{imports_code}\n{source_code}'
        # TODO: validate

        # Execute imports
        exec(imports_code, globals_dict)

        # Perform setup
        exec(f'cudaq.set_target("{target_name}")', globals_dict)
        seed_num = int(request['seed'])
        if seed_num > 0:
            exec(f'cudaq.set_random_seed({seed_num})', globals_dict)

        # Initialize output dictionary
        result = {
            "status": "success",
            "executionContext": {
                "shots": 0,
                "hasConditionalsOnMeasureResults": False
            }
        }
        globals_dict['_json_request_result'] = result

        # Execute main source_code
        simulationStart = int(datetime.now().timestamp() * 1000)
        if target_name == 'nvidia-mgpu' or (
                not cudaq.mpi.is_initialized()) or cudaq.mpi.rank() == 0:
            exec(source_code, globals_dict)
        simulationEnd = int(datetime.now().timestamp() * 1000)

        # Collect results
        result = globals_dict['_json_request_result']
        try:
            cmd_result = subprocess.run(['cudaq-qpud', '--cuda-properties'],
                                        capture_output=True,
                                        text=True)
            deviceProps = json.loads(cmd_result.stdout)
        except:
            deviceProps = dict()

        executionInfo = {
            'requestStart': requestStart,
            'simulationStart': simulationStart,
            'simulationEnd': simulationEnd,
            'deviceProps': deviceProps
        }
        result['executionInfo'] = executionInfo
    except Exception as e:
        error_message = ANSI_PATTERN.sub('', str(e))
        result = {
            'status': 'Failed to process incoming request',
            'errorMessage': error_message
        }
    finally:
        # Only rank 0 prints the result
        if not (cudaq.mpi.is_initialized()) or (cudaq.mpi.rank() == 0):
            with open(jsonFile, 'w') as fp:
                json.dump(result, fp)
                fp.flush()

        if cudaq.mpi.is_initialized():
            cudaq.mpi.finalize()
