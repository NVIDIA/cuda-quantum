# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq
import sys
import json
import pickle
import base64
import threading
import os
import subprocess
from datetime import datetime


def get_deserialized_dict(scoped_dict):
    deserialized_dict = {}

    for key, serialized_bytes in scoped_dict.items():
        try:
            deserialized_value = pickle.loads(serialized_bytes)
            deserialized_dict[key] = deserialized_value
        except (pickle.UnpicklingError, TypeError, Exception) as e:
            print(f"Error deserializing key '{key}': {e}")

    return deserialized_dict


if __name__ == "__main__":
    try:
        watchdog_timeout = int(os.environ.get('WATCHDOG_TIMEOUT_SEC', 0))
        if watchdog_timeout > 0:
            timer = threading.Timer(watchdog_timeout, lambda: os._exit(1))
            timer.start()

        # Read request
        if len(sys.argv) < 2:
            raise (Exception('Too few command-line arguments'))
        jsonFile = sys.argv[1]
        with open(jsonFile, 'rb') as fp:
            request = json.load(fp)

        serialized_ctx = request['serializedCodeExecutionContext']
        imports_code = serialized_ctx['imports']
        source_code = serialized_ctx['source_code']

        serialized_dict = pickle.loads(
            base64.b64decode(serialized_ctx['scoped_var_dict']))
        globals_dict = get_deserialized_dict(serialized_dict)

        # Determine which target to set
        sim2target = {
            'qpp': 'qpp-cpu',
            'custatevec_fp32': 'nvidia',
            'custatevec_fp64': 'nvidia-fp64',
            'tensornet': 'tensornet',
            'tensornet_mps': 'tensornet-mps',
            'dm': 'density-matrix-cpu',
            'nvidia_mgpu': 'nvidia-mgpu'
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

        # We don't have visibility into the difference between requestStart
        # and simulationStart, so simply use simulationStart for both
        executionInfo = {
            'requestStart': simulationStart,
            'simulationStart': simulationStart,
            'simulationEnd': simulationEnd,
            'deviceProps': deviceProps
        }
        result['executionInfo'] = executionInfo

        # Only rank 0 prints the result
        if not (cudaq.mpi.is_initialized()) or (cudaq.mpi.rank() == 0):
            print('\n' + json.dumps(result))
        if watchdog_timeout > 0:
            timer.cancel()  # Must do this before exiting to avoid stall

    except Exception as e:
        result = {
            'status': 'Failed to process incoming request',
            'errorMessage': str(e)
        }
        # Only rank 0 prints the result
        if not (cudaq.mpi.is_initialized()) or (cudaq.mpi.rank() == 0):
            print('\n' + json.dumps(result))
        if watchdog_timeout > 0:
            timer.cancel()  # Must do this before exiting to avoid stall
