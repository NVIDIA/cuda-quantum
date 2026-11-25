# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# Licensed under the Apache License, Version 2.0.
"""Run the C++ outcome program against the existing IQM mock provider.

Requires the mock-service dependencies and the legacy IQM model API (tested
with iqm-client 20.17); no provider account or external endpoint is used.
"""
import json
import multiprocessing
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.request


def serve(repo, port):
    sys.path.insert(0, str(Path(repo) / "python/tests/utils"))
    from mock_qpu.iqm import app, createdJobs
    import iqm.iqm_client as iqm
    import uvicorn

    # The shared mock uses the newer spelling for the same compilation state.
    if not hasattr(iqm.Status, "COMPILATION_STARTED"):
        iqm.Status.COMPILATION_STARTED = iqm.Status.PENDING_COMPILATION

    @app.get("/frequencies-test/jobs")
    def jobs():
        return [{"shots": job.metadata.request.shots,
                 "circuits": [c.model_dump() for c in job.metadata.request.circuits]}
                for job in createdJobs.values()]

    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


def main():
    repo, executable = sys.argv[1:]
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    server = multiprocessing.Process(target=serve, args=(repo, port))
    server.start()
    try:
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            if not server.is_alive():
                raise RuntimeError("IQM mock service failed to start")
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                    break
            except OSError:
                time.sleep(0.05)
        else:
            raise RuntimeError("IQM mock service startup timed out")
        env = dict(os.environ, IQM_SERVER_URL=f"http://127.0.0.1:{port}",
                   IQM_TOKEN="good_access_token", IQM_QC="crystal-20")
        subprocess.run([executable], env=env, check=True, timeout=90)
        with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/frequencies-test/jobs") as response:
            jobs = json.load(response)
        # Zero shots and the unsupported output program must submit no jobs.
        assert [job["shots"] for job in jobs] == [37, 200, 9, 17], jobs
        for job in jobs:
            assert len(job["circuits"]) == 1
            instructions = job["circuits"][0]["instructions"]
            measured = [q for instruction in instructions
                        if instruction["name"] == "measure"
                        for q in instruction["qubits"]]
            assert len(measured) == (1 if job["shots"] == 17 else 2), measured
    finally:
        server.terminate()
        server.join(timeout=5)
        if server.is_alive():
            server.kill()
            server.join()


if __name__ == "__main__":
    main()
