# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Exercise the UDP playback provider against two independent reply sources.

The socket fixture checks transmitted bytes and injects reply faults. The `RTH`
case runs the provider contract checker against validation_app and the UDP bridge. No FPGA or GPU is needed.
"""

import argparse
import os
from pathlib import Path
import re
import signal
import socket
import struct
import subprocess
import tempfile
import threading
import time


def stop(process):
    if process.poll() is None:
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            raise RuntimeError("Consumer did not stop cleanly")


def run(arguments, endpoint):
    spacing_ns = 5000000 if arguments.fixture == "paced" else 120000
    command = [
        arguments.playback, f"--playback-provider={arguments.plugin}",
        f"--endpoint={endpoint}", "--num-messages=31", "--payload-size=37",
        "--timeout-ns=1500000000", "--quiet-ns=200000000",
        f"--spacing-ns={spacing_ns}"
    ]
    return subprocess.run(command, capture_output=True, text=True, timeout=10)


def fixture(arguments):
    # Written by the peer thread and inspected only after it joins.
    errors = []  # Propagate thread assertions back to the test process.
    received = []  # Request IDs observed on the wire, in arrival order.
    arrival_times = []  # Host observations used to check sender pacing.
    # Paced case withholds replies until all requests arrive.
    pending_replies = []
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as server:
        server.bind(("127.0.0.1", 0))
        server.settimeout(0.1)
        done = threading.Event()

        def respond():
            try:
                while not done.is_set():
                    try:
                        data, peer = server.recvfrom(65536)
                    except socket.timeout:
                        continue
                    # Decode the wire independently of the C++ request helpers;
                    # a shared encoding bug must not validate itself.
                    magic, function, length, request_id, stamp = struct.unpack(
                        "<IIIIQ", data[:24])
                    assert len(data) == 96 and magic == 0x43555152
                    assert function == 0x191B6322 and length == 37
                    assert request_id == len(received) + 1
                    assert stamp == 0xFEDCBA9800000000 + request_id
                    payload = bytes(
                        0 if request_id == 1 else 255 if request_id == 2 else
                        (request_id + byte_index * 37) & 255
                        for byte_index in range(37))
                    assert data[24:61] == payload and data[61:] == bytes(35)
                    received.append(request_id)
                    arrival_times.append(time.monotonic())
                    reply = (struct.pack("<IiIIQ", 0x43555153, 0, length,
                                         request_id, stamp) +
                             bytes(
                                 (value + 1) & 255 for value in payload)).ljust(
                                     96, b"\0")
                    if arguments.fixture == "paced":
                        # Withhold every reply until all requests arrive. The
                        # sender must follow spacing without waiting for replies.
                        pending_replies.append((reply, peer))
                        if len(received) == 31:
                            for response, destination in pending_replies:
                                server.sendto(response, destination)
                        continue
                    # Each case changes only the reply behavior; request checks
                    # above still establish that the producer sent correct bytes.
                    if arguments.fixture == "omit" and request_id == 31:
                        continue
                    if arguments.fixture == "corrupt":
                        reply = reply[:24] + bytes([reply[24] ^ 1]) + reply[25:]
                    if arguments.fixture == "delayed" and request_id == 31:
                        time.sleep(
                            0.35)  # Longer than the requested quiet interval.
                    server.sendto(reply, peer)
                    if arguments.fixture == "late-duplicate" and request_id == 31:
                        time.sleep(
                            0.10
                        )  # Must still be observed during the quiet period.
                        server.sendto(reply, peer)
            except Exception as error:
                errors.append(error)

        # Run the peer concurrently with the real playback process. Join it
        # before examining observations or raising assertions from the thread.
        worker = threading.Thread(target=respond)
        worker.start()
        try:
            result = run(arguments,
                         f"port={server.getsockname()[1]} slot_size=96")
        finally:
            done.set()
            worker.join()
        if errors:
            raise errors[0]
        if arguments.fixture in ("delayed", "paced"):
            assert result.returncode == 0 and len(
                received) == 31, result.stdout + result.stderr
            if arguments.fixture == "paced":
                # 30 intervals at 5 ms should span 150 ms. Allow 10 ms of
                # observation slack; an immediate burst cannot pass this check.
                assert arrival_times[-1] - arrival_times[0] >= 0.140
        else:
            # Distinguish missing capture (timeout) from completed but invalid
            # replies; any nonzero exit alone would hide the wrong failure path.
            expected = {
                "omit": "INCOMPLETE: Run deadline",
                "late-duplicate": "duplicate request ID",
                "corrupt": "RPC header/payload mismatch"
            }
            assert result.returncode != 0 and expected[
                arguments.
                fixture] in result.stderr, result.stdout + result.stderr
        print(result.stdout + result.stderr)
        print(
            f"PASS: independent UDP peer, {arguments.fixture}; transmitted RPC headers/payloads checked"
        )


# Match app dispatch and bridge mode explicitly; use four slots to exercise
# wraparound and let the bridge publish an ephemeral listening port.
def consumer(arguments):
    command = [
        arguments.consumer, "--cpu", "--timeout=30",
        f"--bridge={arguments.bridge}"
    ]
    command += ["--", "--port=0", "--num-slots=4", "--slot-size=96"]
    with tempfile.TemporaryDirectory(prefix="playback-udp-") as directory:
        log = Path(directory) / "consumer.log"
        with log.open("w") as output:
            process = subprocess.Popen(command,
                                       stdout=output,
                                       stderr=subprocess.STDOUT,
                                       env=dict(os.environ,
                                                CUDA_VISIBLE_DEVICES=""))
            try:
                deadline = time.monotonic() + 10
                while "=== App Ready ===" not in log.read_text():
                    if process.poll() is not None or time.monotonic(
                    ) >= deadline:
                        raise RuntimeError(log.read_text())
                    time.sleep(0.02)
                # Use the bridge's published endpoint, not a guessed port or
                # socket address from the launching test process.
                endpoint = re.search(r"CUDAQ_VALIDATION_ENDPOINT (.+)",
                                     log.read_text())[1]
                # Two instances check lifecycle and ownership against the real plugin.
                result = subprocess.run([
                    arguments.contract, f"--plugin={arguments.plugin}",
                    f"--endpoint={endpoint}"
                ],
                                        capture_output=True,
                                        text=True,
                                        timeout=15)
                assert result.returncode == 0, result.stdout + result.stderr
                print(result.stdout)
            finally:
                stop(process)
        # Two independent eight-request batches must both reach the dispatcher.
        assert process.returncode == 0 and "Packets dispatched: 16" in log.read_text(
        ), log.read_text()
    print("PASS: UDP provider contracts, GPUs hidden")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--playback")
    parser.add_argument("--plugin", required=True)
    parser.add_argument("--contract")
    parser.add_argument("--consumer")
    parser.add_argument("--bridge")
    parser.add_argument("--fixture",
                        choices=("paced", "corrupt", "omit", "delayed",
                                 "late-duplicate"))
    arguments = parser.parse_args()
    if arguments.fixture:
        fixture(arguments)
    else:
        consumer(arguments)


if __name__ == "__main__":
    main()
