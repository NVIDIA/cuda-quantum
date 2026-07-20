#!/usr/bin/env python3
# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

REWARD_JSON = Path(
    os.environ.get("HARBOR_REWARD_JSON", "/logs/verifier/reward.json"))
REWARD_TXT = Path(
    os.environ.get("HARBOR_REWARD_TXT", "/logs/verifier/reward.txt"))
PROJECT_ROOT = Path(os.environ.get("CUDAQ_HXH_PROJECT",
                                   "/workspace/hxh-plugin"))
CUDAQ_SOURCE = Path(
    os.environ.get("CUDAQ_SOURCE_DIR", "/workspaces/cuda-quantum"))
CUDAQ_BUILD = Path(
    os.environ.get("CUDAQ_BUILD_DIR", "/workspaces/cuda-quantum/build"))
BASE_REVISION = Path("/opt/cudaq-eval-base-revision")
TOOL_CHECKSUMS = Path("/opt/cudaq-eval-tool-sha256")
FIXTURE = Path(os.environ.get("HARBOR_TESTS_DIR", "/tests")) / "input.qke"
COMMAND_TIMEOUT = 60
VERIFIER_BUDGET = 270
VERIFIER_DEADLINE = time.monotonic() + VERIFIER_BUDGET

CHECKS = (
    "configure",
    "build",
    "authored_tests",
    "plugin_load",
    "ir",
    "equivalence",
    "source_isolation",
)


def run(command, *, cwd=None, input_text=None, env=None):
    remaining = VERIFIER_DEADLINE - time.monotonic()
    if remaining <= 0:
        return False, "", "verifier time budget exhausted"
    timeout = min(COMMAND_TIMEOUT, remaining)
    try:
        result = subprocess.run(command,
                                cwd=cwd,
                                input=input_text,
                                text=True,
                                capture_output=True,
                                env=env,
                                check=False,
                                timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, "", f"command timed out after {timeout:.0f}s"
    return result.returncode == 0, result.stdout.strip(), result.stderr.strip()


def detail(stdout, stderr, fallback):
    output = "\n".join(part for part in (stdout, stderr) if part)
    return (output or fallback)[-4000:]


def plugin_candidates(build_root):
    candidates = []
    for suffix in ("*.so", "*.dylib"):
        candidates.extend(build_root.rglob(suffix))
    return sorted(candidates)


def cmake_environment():
    env = os.environ.copy()
    env["CUDAQ_SOURCE_DIR"] = str(CUDAQ_SOURCE)
    env["CUDAQ_BUILD_DIR"] = str(CUDAQ_BUILD)
    prefixes = ["/usr/local/cudaq", str(CUDAQ_BUILD)]
    if env.get("CMAKE_PREFIX_PATH"):
        prefixes.append(env["CMAKE_PREFIX_PATH"])
    env["CMAKE_PREFIX_PATH"] = ";".join(prefixes)
    return env


def environment_is_unchanged():
    if not BASE_REVISION.is_file() or not (CUDAQ_SOURCE / ".git").exists():
        return False, "CUDA-Q baseline revision is unavailable"
    revision = BASE_REVISION.read_text(encoding="utf-8").strip()
    if (CUDAQ_SOURCE / "skills" / "develop-cudaq-pass").exists():
        return False, "baked develop-cudaq-pass skill is agent-visible"
    passed, stdout, stderr = run([
        "/usr/bin/git", "-C",
        str(CUDAQ_SOURCE), "diff", "--quiet", revision, "--", ".",
        ":(exclude)skills/develop-cudaq-pass"
    ])
    if passed:
        passed, stdout, stderr = run([
            "/usr/bin/git", "-C",
            str(CUDAQ_SOURCE), "status", "--porcelain", "--untracked-files=all",
            "--", ".", ":(exclude)skills/develop-cudaq-pass"
        ])
        passed = passed and not stdout
    if not passed:
        return False, detail(stdout, stderr, "CUDA-Q source changed")
    if not TOOL_CHECKSUMS.is_file():
        return False, "CUDA-Q tool checksums are unavailable"
    passed, stdout, stderr = run(
        ["/usr/bin/sha256sum", "--check",
         str(TOOL_CHECKSUMS)])
    return passed, detail(stdout, stderr,
                          "CUDA-Q source and evaluator tools are unchanged")


def reward(checks, reasons):
    qualified = all(checks.values())
    metrics = {
        f"cudaq_hxh_{name}": float(passed) for name, passed in checks.items()
    }
    metrics["cudaq_hxh_qualification"] = float(qualified)
    details = {
        f"cudaq_hxh_{name}": {
            "score": float(passed),
            "reason": reasons[name],
        } for name, passed in checks.items()
    }
    details["cudaq_hxh_qualification"] = {
        "score":
            float(qualified),
        "reason": ("all integration checks passed"
                   if qualified else "one or more integration checks failed"),
    }
    return {
        "overall": float(qualified),
        "custom_metrics": metrics,
        "details": details,
    }


def grade():
    checks = {name: False for name in CHECKS}
    reasons = {name: "prerequisite failed" for name in CHECKS}

    with tempfile.TemporaryDirectory(prefix="cudaq-hxh-grader-") as temporary:
        root = Path(temporary)
        build_root = root / "build"
        env = cmake_environment()

        checks["configure"], stdout, stderr = run([
            "cmake", "-S",
            str(PROJECT_ROOT), "-B",
            str(build_root), "-G", "Ninja"
        ],
                                                  env=env)
        reasons["configure"] = detail(stdout, stderr, "project configured")
        if checks["configure"]:
            checks["build"], stdout, stderr = run(
                ["cmake", "--build", str(build_root)], env=env)
            reasons["build"] = detail(stdout, stderr, "plugin built")

        if checks["build"]:
            listed, stdout, stderr = run(
                ["ctest", "--test-dir",
                 str(build_root), "--show-only=json-v1"],
                env=env)
            try:
                tests = json.loads(stdout).get("tests", [])
            except (AttributeError, json.JSONDecodeError):
                tests = []
            focused = any("hxh" in str(test.get("name", "")).lower()
                          for test in tests
                          if isinstance(test, dict))
            if listed and focused:
                checks["authored_tests"], stdout, stderr = run([
                    "ctest", "--test-dir",
                    str(build_root), "--output-on-failure"
                ],
                                                               env=env)
            reasons["authored_tests"] = detail(stdout, stderr,
                                               "agent-authored tests passed")

        cudaq_opt = CUDAQ_BUILD / "bin" / "cudaq-opt"
        transformed = ""
        attempts = []
        if cudaq_opt.is_file():
            for plugin in plugin_candidates(build_root):
                loaded, stdout, stderr = run([
                    str(cudaq_opt),
                    str(FIXTURE), "--load-cudaq-plugin",
                    str(plugin), "--cudaq-hxh-to-z"
                ],
                                             env=env)
                attempts.append(
                    detail(stdout if not loaded else "", stderr,
                           f"loaded {plugin.name}"))
                if loaded:
                    checks["plugin_load"] = True
                    transformed = stdout
                    break
            reasons["plugin_load"] = attempts[-1] if attempts else (
                "no plugin libraries were produced")
        else:
            reasons["plugin_load"] = "cudaq-opt not found"

        if checks["plugin_load"]:
            checks["ir"], stdout, stderr = run(
                ["/opt/cudaq-eval-tools/FileCheck",
                 str(FIXTURE)],
                input_text=transformed,
                env=env)
            reasons["ir"] = detail(stdout, stderr, "FileCheck passed")

        circuit_check = CUDAQ_BUILD / "bin" / "CircuitCheck"
        if checks["plugin_load"] and circuit_check.is_file():
            checks["equivalence"], stdout, stderr = run(
                [str(circuit_check), str(FIXTURE)],
                input_text=transformed,
                env=env)
            reasons["equivalence"] = detail(stdout, stderr,
                                            "CircuitCheck passed")
        elif not circuit_check.is_file():
            reasons["equivalence"] = "CircuitCheck not found"

        checks["source_isolation"], reasons[
            "source_isolation"] = environment_is_unchanged()

    return reward(checks, reasons)


def main():
    result = grade()
    REWARD_JSON.parent.mkdir(parents=True, exist_ok=True)
    REWARD_JSON.write_text(json.dumps(result, indent=2) + "\n",
                           encoding="utf-8")
    REWARD_TXT.write_text(f"{result['overall']:.1f}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
