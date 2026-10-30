# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Small LQCloud tasks service for backend tests."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import cudaq

TEST_API_KEY = "test-only-fake-lqcloud-key"
SUBMIT = "/api/v1/tasks/async"


@cudaq.kernel
def _execute(n_qubits: int, n_clbits: int, commands: list[int],
             first: list[int], second: list[int],
             angles: list[float]) -> list[bool]:
    q = cudaq.qvector(n_qubits)
    history = [False for _ in range(n_clbits)]
    event = 0
    for i in range(len(commands)):
        if commands[i] == 0:
            h(q[first[i]])
        elif commands[i] == 1:
            x(q[first[i]])
        elif commands[i] == 2:
            y(q[first[i]])
        elif commands[i] == 3:
            rz(angles[i], q[first[i]])
        elif commands[i] == 4:
            z.ctrl(q[first[i]], q[second[i]])
        elif commands[i] == 5:
            reset(q[first[i]])
        else:
            history[event] = mz(q[first[i]])
            event += 1
    return history


def _results(circuit):
    n_qubits = circuit["n_qubits"]
    n_clbits = circuit["n_clbits"]
    shots = circuit["shots"]
    if not isinstance(n_qubits, int) or not 0 < n_qubits <= 8:
        raise ValueError("invalid qubit count")
    if not isinstance(n_clbits, int) or n_clbits <= 0:
        raise ValueError("invalid classical bit count")
    if not isinstance(shots, int) or shots <= 0:
        raise ValueError("invalid shots")

    codes = {"h": 0, "x": 1, "y": 2, "rz": 3, "cz": 4, "reset": 5, "measure": 6}
    commands = []
    first = []
    second = []
    angles = []
    clbit_order = []
    measurements = set()
    measurement_window = set()
    previous_instruction = None
    for instruction in circuit["instructions"]:
        if set(instruction) != {"name", "qubits", "clbits", "params"}:
            raise ValueError("invalid instruction")
        name = instruction["name"]
        qubits = instruction["qubits"]
        clbits = instruction["clbits"]
        params = instruction["params"]
        if any(not isinstance(q, int) or q < 0 or q >= n_qubits
               for q in qubits):
            raise ValueError("invalid qubit")

        if name == "barrier":
            measurement_window.difference_update(qubits)
        elif name in ("h", "x", "y", "reset") and len(qubits) == 1:
            pass
        elif name == "rz" and len(qubits) == 1 and len(params) == 1:
            pass
        elif name == "cz" and len(qubits) == 2:
            pass
        elif name == "x21" and len(qubits) == 1:
            # LQCloud's measure2 readout pulse does not change the logical bit.
            pass
        elif name == "measure" and len(qubits) == 1 and len(clbits) == 1:
            if previous_instruction != ("x21", qubits):
                raise ValueError("measure2 requires x21 on the same qubit")
            clbit = clbits[0]
            if not isinstance(clbit, int) or clbit < 0 or clbit >= n_clbits:
                raise ValueError("invalid classical bit")
            if clbit in measurements:
                raise ValueError("classical bit already recorded")
            if qubits[0] in measurement_window:
                raise ValueError("repeated measurement requires a barrier")
            measurements.add(clbit)
            measurement_window.add(qubits[0])
            clbit_order.append(clbit)
        else:
            raise ValueError("unsupported instruction")
        previous_instruction = (name, qubits)
        if name in codes:
            commands.append(codes[name])
            first.append(qubits[0])
            second.append(qubits[-1])
            angles.append(float(params[0]) if name == "rz" else 0.0)

    if len(measurements) != n_clbits:
        raise ValueError("incomplete measurement")

    counts = {}
    memory = []
    for record in cudaq.run(_execute,
                            n_qubits,
                            n_clbits,
                            commands,
                            first,
                            second,
                            angles,
                            shots_count=shots):
        if len(record) != n_clbits:
            raise ValueError("incomplete measurement history")
        bits = ["0"] * n_clbits
        for clbit, bit in zip(clbit_order, record):
            bits[clbit] = str(int(bit))
        key = "".join(bits)
        counts[key] = counts.get(key, 0) + 1
        memory.append(key)
    return counts, memory


class _Handler(BaseHTTPRequestHandler):

    def log_message(self, *_):
        pass

    def send_json(self, status, value):
        body = json.dumps(value).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def authenticated(self):
        return self.headers.get("X-API-Key") == TEST_API_KEY

    def do_POST(self):
        try:
            if self.path != SUBMIT or not self.authenticated():
                raise ValueError("invalid request")
            body = json.loads(
                self.rfile.read(int(self.headers.get("Content-Length", 0))))
            if body.get("qpu_name") != "mock_qpu":
                raise ValueError("invalid QPU")
            command = body["command"]
            if command.get("action") != "run_circuit":
                raise ValueError("invalid action")
            circuit = command["circuit"]
            result_format = circuit.get("result_format")
            if result_format not in ("counts", "memory"):
                raise ValueError("invalid result format")
            with self.server.task_lock:
                counts, memory = _results(circuit)
            result = {"counts": counts}
            if result_format == "memory":
                result = {
                    "result_format": "memory",
                    "shots": circuit["shots"],
                    "memory": memory,
                }
            with self.server.task_lock:
                self.server.next_task += 1
                task_id = f"task_{self.server.next_task:016x}"
                self.server.tasks[task_id] = result
            self.send_json(200, {"task_id": task_id})
        except (ValueError, KeyError, TypeError, json.JSONDecodeError):
            self.send_json(400, {"error": "invalid request"})

    def do_GET(self):
        if not self.authenticated():
            return self.send_json(401, {"error": "missing test key"})
        if self.path == "/api/v1/qpus":
            return self.send_json(200, [{
                "name": "mock_qpu",
                "qubits": 3,
                "status": "active",
                "supported_measurement_types": ["measure2"],
                "topology": {
                    "type": "chain",
                    "coupling_map": [[0, 1], [1, 2]],
                },
            }])
        prefix = SUBMIT + "/"
        if not self.path.startswith(prefix):
            return self.send_json(404, {"error": "unknown task"})
        task_id = self.path[len(prefix):]
        with self.server.task_lock:
            result = self.server.tasks.get(task_id)
        if result is None:
            return self.send_json(404, {"error": "unknown task"})
        self.send_json(200, {
            "task_id": task_id,
            "status": "completed",
            "result": result,
        })


def startServer(port):
    cudaq.set_target("qpp-cpu")
    with ThreadingHTTPServer(("localhost", port), _Handler) as server:
        server.next_task = 0
        server.tasks = {}
        server.task_lock = threading.Lock()
        server.serve_forever()
