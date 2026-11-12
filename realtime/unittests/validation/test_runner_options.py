# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Check argument forwarding at the shared/legacy runner boundary."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

parser = argparse.ArgumentParser()
parser.add_argument("--runner", required=True)
parser.add_argument("--legacy-runner", required=True)
arguments = parser.parse_args()
with tempfile.TemporaryDirectory(prefix="validation options ") as temporary:
    root = Path(temporary)
    runner = root / "validation_test.sh"
    shutil.copyfile(arguments.runner, runner)

    # Record exact argument boundaries, then provide the minimal ready/start
    # handshake. No plugins or hardware are loaded by these shell stubs.
    def recorder(name):
        return ("#!/usr/bin/env bash\n"
                "printf '%s\\0' \"$@\" > \"$TEST_ARGS_DIR/" + name +
                ".args\"\n")

    rth = root / "validation_app"
    rth.write_text(
        recorder("rth") + "trap 'exit 0' INT TERM\n"
        "echo 'CUDAQ_VALIDATION_ENDPOINT fixture'\n"
        "echo '=== App Ready ==='\n"
        "while true; do sleep 0.01; done\n")
    playback = root / "playback_app"
    playback.write_text(
        recorder("playback") + "echo CUDAQ_PLAYBACK_READY\n"
        'for option in "$@"; do case "$option" in --start-file=*) gate=${option#*=};; esac; done\n'
        'until [[ -f "$gate" ]]; do sleep 0.01; done\n')
    legacy = root / "roce_test.sh"
    legacy.write_text(recorder("legacy"))
    for executable in (rth, playback):
        executable.chmod(0o755)
    sequence = 0

    def run(options, legacy_mode=False, expected_status=0):
        global sequence
        sequence += 1
        case = root / str(sequence)
        case.mkdir()
        command = [
            "bash",
            str(runner), f"--validation-app={rth}", f"--playback-app={playback}"
        ]
        if not legacy_mode:
            command += [
                "--bridge=libcudaq-realtime-bridge-gpu-roce.so",
                "--playback-provider=libcudaq-playback-provider-hsb.so",
                f"--output-dir={case / 'logs'}"
            ]
        result = subprocess.run(command + options,
                                capture_output=True,
                                text=True,
                                env=dict(os.environ, TEST_ARGS_DIR=str(case)),
                                timeout=10)
        assert result.returncode == expected_status, result.stdout + result.stderr
        received = {}
        for name in ("rth", "playback", "legacy"):
            record = case / f"{name}.args"
            received[name] = record.read_text().split(
                "\0")[:-1] if record.exists() else []
        return received

    # Invalid flags must fail before delegation can run network setup. The
    # legacy executable here is only a recorder; these checks never touch a NIC.
    for flag in ("--help", "-h", "--roce", "--udp", "--cpu", "--unified",
                 "--forward", "--save-temps", "--build", "--setup-network",
                 "--no-run", "--emulate", "--no-verify"):
        for value in ("false", "no", "true", ""):
            received = run(["--setup-network", f"{flag}={value}"],
                           True,
                           expected_status=1)
            assert not any(received.values())
    for option in ("--gpu=false", "--gpu=", "--gpu=2147483648", "--gpu-id=no"):
        received = run(["--setup-network", option], True, expected_status=1)
        assert not any(received.values())

    # Preserve the default FPGA ring and workload, which must not change when
    # plugin selection or argument forwarding is refactored.
    received = run([])
    for option in ("--gpu", "--dispatch=separate", "--page-size=384",
                   "--num-pages=128"):
        assert option in received["rth"]
    for option in ("--num-messages=100", "--message-size=32",
                   "--spacing-ns=120000", "--quiet-ns=1000000000",
                   "--timeout-ns=30000000000"):
        assert option in received["playback"]

    # Quoted lists preserve spaces and literal shell expressions without executing them.
    marker = root / "must-not-exist"
    literal = f"$(touch '{marker}')"
    received = run([
        '--bridge-arg=--label="two words" --device=mlx5_4',
        f'--playback-arg=--literal="{literal}" --empty=""'
    ])
    assert received["rth"][-2:] == ["--label=two words", "--device=mlx5_4"]
    assert received["playback"][-2:] == [f"--literal={literal}", "--empty="]
    assert not marker.exists()
    received = run([
        "--gpu", "1", "--bridge-arg", "--device mlx5_4", "--playback-arg",
        "--emulate=false"
    ])
    assert "--gpu-id=1" in received["rth"]
    assert received["rth"][-1] == "--device=mlx5_4"
    assert received["playback"][-1] == "--emulate=false"

    # Both built-in RoCE presets must honor the explicit device in either
    # spelling, without retaining an automatically selected device alongside it.
    for execution in ("cpu", "gpu"):
        for options in (["--bridge-arg=--device mlx5_4"
                        ], ["--bridge-arg=--device=mlx5_4"],
                        ["--device", "mlx5_4"], ["--", "--device", "mlx5_4"]):
            received = run([
                f"--{execution}",
                f"--bridge=libcudaq-realtime-bridge-{execution}-roce.so",
                *options
            ])
            assert [
                option for option in received["rth"]
                if option.startswith("--device")
            ] == ["--device=mlx5_4"]
    for options in ("--device", "--device=", "--device --peer-ip=192.168.0.2"):
        received = run([f"--bridge-arg={options}"], expected_status=1)
        assert not received["rth"] and not received["playback"]
    received = run(["--bridge=custom.so", "--bridge-arg=--device mlx5_4"])
    assert received["rth"][-2:] == ["--device", "mlx5_4"]

    # The legacy parser needs --bin-dir before -- and unchanged bridge options after it.
    bridge_options = [
        "--device=mlx5_4", "--label=two words", "--",
        "--output-dir=bridge-value"
    ]
    received = run(
        ["--app=validation_app", "--bridge=custom.so", "--", *bridge_options],
        True)
    assert received["legacy"] == [
        "--app", "validation_app", "--bridge", "custom.so", "--bin-dir",
        str(root), "--", *bridge_options
    ]
    received = run([
        "--emulate", "--dispatch=unified", "--message-size=40", "--timeout", "1"
    ], True)
    assert received["legacy"] == [
        "--emulate", "--unified", "--payload-size", "16", "--timeout", "1",
        "--bin-dir",
        str(root)
    ]
    received = run(["--emulate", "--spacing-ns=1000"], True, expected_status=1)
    assert not received["legacy"]

    # Exercise the real legacy parser without starting networking or playback.
    shutil.copyfile(arguments.legacy_runner, legacy)
    run([
        "--emulate", "--no-run", "--dispatch=unified", "--message-size=40",
        "--timeout", "1"
    ], True)
print("PASS: default workload, quoted options and legacy argument forwarding")
