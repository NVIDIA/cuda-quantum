# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def _load_benchmark_module():
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "benchmarks" / "mklq" / "bench_mklq_targets.py"
    spec = importlib.util.spec_from_file_location("bench_mklq_targets", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_probability_benchmark_module():
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "benchmarks" / "mklq" / "bench_probability_kernels.py"
    spec = importlib.util.spec_from_file_location("bench_probability_kernels",
                                                  script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_summary_renderer_module():
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "benchmarks" / "mklq" / "summarize_reports.py"
    spec = importlib.util.spec_from_file_location("summarize_reports", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_summary_generator_module():
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "benchmarks" / "mklq" / "make_summary.py"
    spec = importlib.util.spec_from_file_location("make_summary", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_clean_benchmark_gate_module():
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "benchmarks" / "mklq" / "run_clean_cpu_benchmark.py"
    spec = importlib.util.spec_from_file_location("run_clean_cpu_benchmark",
                                                  script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_correctness_gate_module():
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "benchmarks" / "mklq" / "run_correctness_gate.py"
    spec = importlib.util.spec_from_file_location("run_correctness_gate",
                                                  script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _raw_benchmark_report(dirty=False, cases=None, results=None):
    cases = cases or ["y-state"]
    results = results or []
    return {
        "schema_version": "mklq-benchmark-v1",
        "machine": {
            "cpu_brand": "Apple M5",
            "logical_cores": 10,
            "memory_bytes": 17179869184,
            "macos_version": "26.5.1",
        },
        "provenance": {
            "git": {
                "branch": "main",
                "commit": "abc123",
                "dirty": dirty,
                "root": "/repo",
                "status_short": [" M file"] if dirty else [],
            },
            "environment": {
                "OMP_NUM_THREADS": "10",
                "OMP_PROC_BIND": "close",
            },
        },
        "config": {
            "targets": ["qpp-cpu", "mklq-cpu"],
            "cases": cases,
            "qubits": [20],
            "shots": 1024,
            "shot_counts": sorted({
                row.get("shots", 1024)
                for row in results
            } or {1024}),
            "repeats": 2,
            "warmups": 1,
            "layers": 8,
            "isolate_rows": True,
            "command": ["bench_mklq_targets.py", "--isolate-rows"],
        },
        "results": results,
    }


def _benchmark_row(target, case, elapsed, shots=1024):
    return {
        "target": target,
        "case": case,
        "qubits": 20,
        "shots": shots,
        "status": "ok",
        "estimated_state_bytes": 16777216,
        "repeats": 2,
        "warmups": 1,
        "isolated_process": {
            "runtime": {
                "cudaq_module_file": "/tmp/cudaq/__init__.py",
                "cudaq_version": "test-version",
                "module_from_build_tree": False,
                "python_prefix": "/tmp/python",
            }
        },
        "metrics": {
            "elapsed_seconds_median": elapsed,
            "elapsed_seconds_min": elapsed,
            "elapsed_seconds_max": elapsed,
            "process_max_rss_bytes_cumulative": 1234,
        },
    }


def test_mklq_summary_generator_builds_sanitized_summary(tmp_path):
    module = _load_summary_generator_module()
    gate_rows = [
        _benchmark_row("qpp-cpu", "y-state", 10.0),
        _benchmark_row("mklq-cpu", "y-state", 2.0),
    ]
    sampling_rows = [
        _benchmark_row("qpp-cpu", "sample-full-register", 8.0, shots=1024),
        _benchmark_row("mklq-cpu", "sample-full-register", 1.0, shots=1024),
        _benchmark_row("qpp-cpu", "sample-full-register", 30.0, shots=65536),
        _benchmark_row("mklq-cpu", "sample-full-register", 3.0, shots=65536),
    ]
    gate_path = tmp_path / "gate.json"
    sampling_path = tmp_path / "sampling.json"
    gate_path.write_text(json.dumps(
        _raw_benchmark_report(cases=["y-state"], results=gate_rows)),
                         encoding="utf-8")
    sampling_path.write_text(json.dumps(
        _raw_benchmark_report(cases=["sample-full-register"],
                              results=sampling_rows)),
                             encoding="utf-8")

    summary = module.build_summary(
        raw_paths=[gate_path, sampling_path],
        summary_id="local-clean-test",
        evidence_kind="clean_local_benchmark_evidence",
        reference_target="qpp-cpu",
        candidate_target="mklq-cpu",
        ratio_group="clean_worktree_cross_target_ratio",
        performance_scope="local test only",
        summary_text="Synthetic clean benchmark summary.",
        runtime_note="synthetic runtime note",
    )

    assert summary["schema_version"] == module.SUMMARY_SCHEMA_VERSION
    assert summary["evidence_kind"] == "clean_local_benchmark_evidence"
    assert summary["git"]["dirty"] is False
    assert summary["interpretation"]["clean_worktree"] is True
    assert summary["interpretation"]["runtime_build_note"] == (
        "synthetic runtime note")
    assert summary["raw_results"][0]["status_rows"] == {"ok": 2}
    assert summary["raw_results"][0]["sha256"] == module.sha256_file(
        gate_path)
    assert summary["raw_results"][1]["status_rows"] == {"ok": 4}
    assert summary["config"]["targets"] == ["qpp-cpu", "mklq-cpu"]
    assert summary["config"]["cases"] == [
        "y-state", "sample-full-register"
    ]
    assert summary["config"]["shot_counts"] == [1024, 65536]

    assert len(summary["rows"]) == 6
    assert "isolated_process" not in summary["rows"][0]
    ratios = summary["comparison"]["clean_worktree_cross_target_ratio"]
    assert ratios["qpp_cpu_over_mklq_cpu_y_state_q20"] == 5.0
    assert ratios[
        "qpp_cpu_over_mklq_cpu_sample_full_register_q20_65536_shots"
    ] == 10.0
    elapsed = summary["comparison"]["mklq_cpu_elapsed_seconds_median"]
    assert elapsed["sample_full_register_q20_1024_shots"] == 1.0


def test_mklq_summary_generator_rejects_dirty_by_default(tmp_path):
    module = _load_summary_generator_module()
    raw_path = tmp_path / "dirty.json"
    raw_path.write_text(json.dumps(
        _raw_benchmark_report(dirty=True,
                              results=[_benchmark_row("qpp-cpu", "y-state",
                                                      1.0)])),
                        encoding="utf-8")

    with pytest.raises(ValueError, match="dirty git worktree"):
        module.build_summary(raw_paths=[raw_path],
                             summary_id="dirty",
                             evidence_kind="clean_local_benchmark_evidence",
                             reference_target="qpp-cpu",
                             candidate_target="mklq-cpu",
                             ratio_group=None,
                             performance_scope="local",
                             summary_text="dirty")


def test_mklq_clean_cpu_gate_plan_uses_fixed_environment(tmp_path):
    module = _load_clean_benchmark_gate_module()
    config = module.GateConfig(
        repo_root=tmp_path,
        pythonpath="/tmp/cudaq-runtime",
        stamp="2026-06-21",
        qubits=20,
        threads=10,
        repeats=2,
        warmups=1,
        layers=8,
        shots=1024,
        shot_counts="1024,65536",
        results_dir=tmp_path / "results",
        reports_dir=tmp_path / "reports",
        evidence_output=tmp_path / "benchmark-evidence.md",
        targets="qpp-cpu,mklq-cpu",
        gate_cases="y-state,cy-state",
        sampling_cases="sample-full-register,sample-partial-register",
        summary_id="local-clean-cpu-q20-2026-06-21",
        evidence_kind="clean_local_benchmark_evidence",
        ratio_group="clean_worktree_cross_target_ratio",
        performance_scope="local test only",
        summary_text="Synthetic clean benchmark gate.",
        runtime_note="synthetic runtime note",
        allow_dirty=False,
        skip_benchmark=False,
        refresh_evidence=True,
    )

    plan = module.build_plan(config)

    assert plan["environment"] == {
        "OMP_NUM_THREADS": "10",
        "OMP_PROC_BIND": "close",
        "OMP_DYNAMIC": "false",
        "VECLIB_MAXIMUM_THREADS": "1",
        "PYTHONPATH": "/tmp/cudaq-runtime",
    }
    assert plan["paths"]["gate_raw"].endswith(
        "local-clean-cpu-gate-y-cy-q20-2026-06-21.json")
    assert plan["paths"]["sampling_raw"].endswith(
        "local-clean-cpu-sampling-q20-2026-06-21.json")
    assert plan["paths"]["summary"].endswith(
        "local-clean-cpu-q20-2026-06-21.summary.json")
    gate_command = plan["commands"]["gate_raw"]
    assert "--isolate-rows" in gate_command
    assert gate_command[gate_command.index("--cases") + 1] == (
        "y-state,cy-state")
    assert gate_command[gate_command.index("--targets") + 1] == (
        "qpp-cpu,mklq-cpu")
    sampling_command = plan["commands"]["sampling_raw"]
    assert sampling_command[sampling_command.index("--shot-counts") +
                            1] == "1024,65536"
    summary_command = plan["commands"]["summary"]
    assert "--allow-dirty" not in summary_command
    assert summary_command[summary_command.index("--ratio-group") + 1] == (
        "clean_worktree_cross_target_ratio")


def test_mklq_clean_cpu_gate_skip_benchmark_runs_summary_only(monkeypatch,
                                                              tmp_path):
    module = _load_clean_benchmark_gate_module()
    config = module.GateConfig(
        repo_root=tmp_path,
        pythonpath="/tmp/cudaq-runtime",
        stamp="2026-06-21",
        qubits=20,
        threads=10,
        repeats=2,
        warmups=1,
        layers=8,
        shots=1024,
        shot_counts="1024,65536",
        results_dir=tmp_path / "results",
        reports_dir=tmp_path / "reports",
        evidence_output=tmp_path / "benchmark-evidence.md",
        targets="qpp-cpu,mklq-cpu",
        gate_cases="y-state,cy-state",
        sampling_cases="sample-full-register,sample-partial-register",
        summary_id="local-clean-cpu-q20-2026-06-21",
        evidence_kind="clean_local_benchmark_evidence",
        ratio_group="clean_worktree_cross_target_ratio",
        performance_scope="local test only",
        summary_text="Synthetic clean benchmark gate.",
        runtime_note="synthetic runtime note",
        allow_dirty=False,
        skip_benchmark=True,
        refresh_evidence=True,
    )
    calls = []

    def fake_run_command(command, env_overlay, cwd):
        calls.append((command, env_overlay, cwd))

    monkeypatch.setattr(module, "run_command", fake_run_command)

    module.run_gate(config, plan_only=False)

    assert len(calls) == 2
    assert calls[0][0] == module.build_plan(config)["commands"]["summary"]
    assert calls[1][0] == module.build_plan(config)["commands"]["evidence"]
    assert calls[0][1]["OMP_NUM_THREADS"] == "10"
    assert calls[0][2] == tmp_path


def _correctness_gate_config(module, tmp_path):
    return module.CorrectnessGateConfig(
        repo_root=tmp_path,
        pythonpath="/tmp/cudaq-runtime",
        nvqpp=Path("/tmp/cudaq-runtime/bin/nvq++"),
        build_dir=tmp_path / "build-python",
        output=tmp_path / "correctness-gate.json",
        stamp="2026-06-21",
        python_executable="/usr/bin/python3",
        timeout_seconds=123,
        tail_chars=80,
        skip_python=False,
        skip_nvqpp=False,
        skip_ctest=False,
    )


def test_mklq_correctness_gate_plan_lists_expected_steps(tmp_path):
    module = _load_correctness_gate_module()
    config = _correctness_gate_config(module, tmp_path)

    plan = module.build_plan(config)

    assert plan["schema_version"] == "mklq-correctness-gate-v1"
    assert plan["environment"] == {
        "PYTHONPATH": "/tmp/cudaq-runtime",
        "CUDAQ_NVQPP": "/tmp/cudaq-runtime/bin/nvq++",
    }
    assert [step["name"] for step in plan["steps"]] == [
        "python_target_smoke",
        "nvqpp_smoke",
        "target_config_ctest",
    ]
    assert "python/tests/backends/test_mklq_python_api.py" in plan["steps"][0][
        "command"]
    assert "python/tests/backends/test_mklq_cpu_correctness_fixtures.py" in plan[
        "steps"][0]["command"]
    assert "python/tests/backends/test_mklq_metal_correctness_fixtures.py" in plan[
        "steps"][0]["command"]
    assert "python/tests/builder/test_mklq_targets.py" in plan["steps"][0][
        "command"]
    assert "python/tests/backends/test_mklq_nvqpp_smoke.py" in plan["steps"][
        1]["command"]
    assert plan["steps"][2]["command"][0] == "ctest"
    assert plan["steps"][2]["command"][plan["steps"][2]["command"].index("-R") +
                                      1] == module.TARGET_CONFIG_REGEX


def test_mklq_correctness_gate_writes_json_summary(monkeypatch, tmp_path):
    module = _load_correctness_gate_module()
    config = _correctness_gate_config(module, tmp_path)
    calls = []

    def fake_run(command, cwd, env, capture_output, text, timeout):
        calls.append({
            "command": command,
            "cwd": cwd,
            "env": env,
            "capture_output": capture_output,
            "text": text,
            "timeout": timeout,
        })
        return subprocess.CompletedProcess(command,
                                           0,
                                           stdout="ok\n",
                                           stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(module, "git_snapshot",
                        lambda root: {
                            "root": str(root),
                            "branch": "test",
                            "commit": "abc123",
                            "dirty": False,
                            "status_short": [],
                        })

    report = module.run_gate(config, plan_only=False)
    written = json.loads(config.output.read_text(encoding="utf-8"))

    assert report["summary"] == {
        "status": "passed",
        "passed": 3,
        "failed": 0,
        "skipped": 0,
    }
    assert written["summary"] == report["summary"]
    assert [call["command"] for call in calls] == [
        step["command"] for step in module.build_plan(config)["steps"]
    ]
    assert calls[0]["env"]["PYTHONPATH"] == "/tmp/cudaq-runtime"
    assert calls[1]["env"]["CUDAQ_NVQPP"] == "/tmp/cudaq-runtime/bin/nvq++"
    assert all(step["returncode"] == 0 for step in written["steps"])
    assert all(step["stdout_tail"] == "ok\n" for step in written["steps"])


def test_mklq_summary_renderer_builds_stable_markdown(tmp_path):
    module = _load_summary_renderer_module()
    common = {
        "schema_version": module.SUMMARY_SCHEMA_VERSION,
        "evidence_kind": "local_tuning_evidence",
        "machine": {
            "cpu_brand": "Apple M5",
            "logical_cores": 10,
            "memory_bytes": 17179869184,
            "macos_version": "26.5.1",
        },
        "config": {
            "targets": ["qpp-cpu", "mklq-cpu"],
            "cases": ["sample-full-register"],
            "qubits": [20],
            "shots": 1024,
            "repeats": 2,
            "warmups": 1,
            "layers": 8,
            "isolate_rows": True,
        },
        "rows": [{
            "status": "ok"
        }, {
            "status": "error"
        }],
        "raw_results": [{
            "path": "benchmarks/mklq/results/local-a.json",
            "sha256": "abcdef1234567890abcdef1234567890abcdef1234567890",
        }],
        "comparison": {
            "same_day_ratio": {
                "qpp_cpu_over_mklq_cpu": 2.5
            },
            "probe_seconds": 0.125,
        },
        "interpretation": {
            "do_not_treat_as_clean_release_provenance": True
        },
    }

    z_summary = dict(common)
    z_summary["summary_id"] = "z-summary"
    a_summary = dict(common)
    a_summary["summary_id"] = "a-summary"

    z_path = tmp_path / "z.summary.json"
    a_path = tmp_path / "a.summary.json"
    z_path.write_text(json.dumps(z_summary), encoding="utf-8")
    a_path.write_text(json.dumps(a_summary), encoding="utf-8")

    digests = module.load_digests([z_path, a_path])
    assert [digest["summary_id"] for digest in digests] == [
        "a-summary", "z-summary"
    ]

    markdown = module.render_markdown(digests)
    assert markdown.index("a-summary") < markdown.index("z-summary")
    assert "local benchmark evidence" in markdown
    assert "Apple M5, 10 logical cores, 16 GiB RAM, macOS 26.5.1" in markdown
    assert (
        "shots=1024; repeats=2; warmups=1; layers=8; isolate_rows=true"
        in markdown)
    assert "error=1, ok=1" in markdown
    assert "sha256=abcdef123456" in markdown
    assert "`same_day_ratio.qpp_cpu_over_mklq_cpu` | 2.50x" in markdown
    assert "`probe_seconds` | 0.125 s" in markdown


def test_mklq_summary_renderer_rejects_unexpected_schema(tmp_path):
    module = _load_summary_renderer_module()
    summary_path = tmp_path / "bad.summary.json"
    summary_path.write_text(json.dumps({"schema_version": "other"}),
                            encoding="utf-8")

    with pytest.raises(ValueError, match="expected"):
        module.load_summary(summary_path)


def test_mklq_benchmark_dry_run_writes_schema(tmp_path):
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "benchmarks" / "mklq" / "bench_mklq_targets.py"
    output = tmp_path / "dry-run.json"
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["OMP_NUM_THREADS"] = "3"

    subprocess.run([
        sys.executable,
        str(script),
        "--dry-run",
        "--targets",
        "qpp-cpu,mklq-cpu",
        "--cases",
        "gate-state,sample-ghz",
        "--qubits",
        "2,3",
        "--shots",
        "8",
        "--repeats",
        "1",
        "--output",
        str(output),
    ],
                   check=True,
                   capture_output=True,
                   text=True,
                   env=env)

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["schema_version"] == "mklq-benchmark-v1"
    assert report["machine"]["platform"]
    assert report["provenance"]["cwd"]
    assert report["provenance"]["git"]["commit"]
    assert report["provenance"]["environment"]["OMP_NUM_THREADS"] == "3"
    assert "mklq-metal" in report["target_notes"]
    assert "mixed-path" in report["target_notes"]["mklq-metal"]
    assert "resident" in report["target_notes"]["mklq-metal"]
    assert "probability-fill" in report["target_notes"]["mklq-metal"]
    assert "marginal probability" in report["target_notes"]["mklq-metal"]
    assert "measurement-collapse" in report["target_notes"]["mklq-metal"]
    assert "CPU-oracle fallback" in report["target_notes"]["mklq-metal"]
    assert "host-side" in report["target_notes"]["mklq-metal"]
    assert "mklq_metal" in report["target_notes"]["mklq-metal"]
    assert report["config"]["targets"] == ["qpp-cpu", "mklq-cpu"]
    assert report["config"]["cases"] == ["gate-state", "sample-ghz"]

    rows = report["results"]
    assert len(rows) == 8
    assert {row["status"] for row in rows} == {"planned"}
    assert {row["target"] for row in rows} == {"qpp-cpu", "mklq-cpu"}
    assert {row["case"] for row in rows} == {"gate-state", "sample-ghz"}
    assert {row["qubits"] for row in rows} == {2, 3}

    for row in rows:
        assert row["shots"] == 8
        assert row["repeats"] == 1
        assert row["estimated_state_bytes"] == 16 * (1 << row["qubits"])
        assert row["metrics"] == {}


def test_mklq_benchmark_default_targets_are_apple_silicon_gated():
    module = _load_benchmark_module()

    assert module.default_targets_for_platform("Darwin", "arm64") == [
        "qpp-cpu", "mklq-cpu", "mklq-metal"
    ]
    assert module.default_targets_for_platform("Darwin", "x86_64") == [
        "qpp-cpu"
    ]
    assert module.default_targets_for_platform("Linux", "aarch64") == [
        "qpp-cpu"
    ]


def test_mklq_benchmark_returns_nonzero_on_row_error(tmp_path):
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "benchmarks" / "mklq" / "bench_mklq_targets.py"
    output = tmp_path / "error.json"

    result = subprocess.run([
        sys.executable,
        str(script),
        "--targets",
        "not-a-target",
        "--cases",
        "gate-state",
        "--qubits",
        "1",
        "--repeats",
        "1",
        "--warmups",
        "1",
        "--layers",
        "1",
        "--output",
        str(output),
    ],
                            capture_output=True,
                            text=True)

    assert result.returncode == 1
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["runtime"]["cudaq_module_file"]
    assert "PYTHONPATH" in report["provenance"]["environment"]
    assert report["results"][0]["status"] == "error"
    assert "not-a-target" in report["results"][0]["error"]

    allowed = subprocess.run([
        sys.executable,
        str(script),
        "--allow-errors",
        "--targets",
        "not-a-target",
        "--cases",
        "gate-state",
        "--qubits",
        "1",
        "--repeats",
        "1",
        "--warmups",
        "1",
        "--layers",
        "1",
        "--output",
        str(output),
    ],
                             capture_output=True,
                             text=True)

    assert allowed.returncode == 0


def test_mklq_benchmark_dry_run_records_row_isolation_flag(tmp_path):
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "benchmarks" / "mklq" / "bench_mklq_targets.py"
    output = tmp_path / "dry-run-isolated.json"
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)

    subprocess.run([
        sys.executable,
        str(script),
        "--dry-run",
        "--isolate-rows",
        "--targets",
        "mklq-cpu",
        "--cases",
        "sample-ghz",
        "--qubits",
        "3",
        "--output",
        str(output),
    ],
                   check=True,
                   capture_output=True,
                   text=True,
                   env=env)

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["config"]["isolate_rows"] is True
    assert len(report["results"]) == 1
    assert report["results"][0]["status"] == "planned"


def test_mklq_benchmark_dry_run_accepts_single_qubit_case(tmp_path):
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "benchmarks" / "mklq" / "bench_mklq_targets.py"
    output = tmp_path / "dry-run-single-qubit.json"
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)

    subprocess.run([
        sys.executable,
        str(script),
        "--dry-run",
        "--targets",
        "mklq-cpu",
        "--cases",
        "single-qubit-state",
        "--qubits",
        "3",
        "--output",
        str(output),
    ],
                   check=True,
                   capture_output=True,
                   text=True,
                   env=env)

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["config"]["cases"] == ["single-qubit-state"]
    rows = report["results"]
    assert len(rows) == 1
    assert rows[0]["status"] == "planned"
    assert rows[0]["case"] == "single-qubit-state"
    assert rows[0]["estimated_state_bytes"] == 16 * (1 << 3)


def test_mklq_benchmark_dry_run_accepts_single_gate_cases(tmp_path):
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "benchmarks" / "mklq" / "bench_mklq_targets.py"
    output = tmp_path / "dry-run-single-gate.json"
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)

    subprocess.run([
        sys.executable,
        str(script),
        "--dry-run",
        "--targets",
        "mklq-cpu",
        "--cases",
        "h-state,y-state,rx-state,ry-state,rz-state",
        "--qubits",
        "3",
        "--output",
        str(output),
    ],
                   check=True,
                   capture_output=True,
                   text=True,
                   env=env)

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["config"]["cases"] == [
        "h-state",
        "y-state",
        "rx-state",
        "ry-state",
        "rz-state",
    ]
    rows = report["results"]
    assert len(rows) == 5
    assert {row["status"] for row in rows} == {"planned"}
    assert {row["case"] for row in rows} == {
        "h-state",
        "y-state",
        "rx-state",
        "ry-state",
        "rz-state",
    }
    assert {row["estimated_state_bytes"] for row in rows} == {16 * (1 << 3)}


def test_mklq_benchmark_single_gate_cases_record_gate_specific_metrics(
        monkeypatch):
    module = _load_benchmark_module()

    class FakeKernel:

        def __init__(self):
            self.operations = []

        def qalloc(self, qubits):
            return list(range(qubits))

        def h(self, target):
            self.operations.append(("h", target))

        def y(self, target):
            self.operations.append(("y", target))

        def rx(self, theta, target):
            self.operations.append(("rx", theta, target))

        def ry(self, theta, target):
            self.operations.append(("ry", theta, target))

        def rz(self, theta, target):
            self.operations.append(("rz", theta, target))

    class FakeCudaq:

        def __init__(self):
            self.kernels = []

        def reset_target(self):
            pass

        def set_target(self, target):
            assert target == "mklq-cpu"

        def set_random_seed(self, seed):
            assert seed == 13

        def make_kernel(self):
            kernel = FakeKernel()
            self.kernels.append(kernel)
            return kernel

        def get_state(self, kernel):
            return object()

    def fake_timed_repeats(action, repeats):
        assert repeats == 1
        action()
        return [0.25]

    monkeypatch.setattr(module, "timed_repeats", fake_timed_repeats)
    monkeypatch.setattr(module, "process_max_rss_bytes", lambda: 4096)

    cases = {
        "h-state": ("h_gate_count", "h_gate_state_throughput_per_second"),
        "y-state": ("y_gate_count", "y_gate_state_throughput_per_second"),
        "rx-state": ("rx_gate_count", "rx_gate_state_throughput_per_second"),
        "ry-state": ("ry_gate_count", "ry_gate_state_throughput_per_second"),
        "rz-state": ("rz_gate_count", "rz_gate_state_throughput_per_second"),
    }

    for case, (count_key, throughput_key) in cases.items():
        fake_cudaq = FakeCudaq()
        row = module.run_case(fake_cudaq,
                              "mklq-cpu",
                              case,
                              qubits=3,
                              shots=16,
                              repeats=1,
                              warmups=0,
                              layers=2)

        assert row["status"] == "ok"
        metrics = row["metrics"]
        assert metrics["state_prep_gate_count"] == 6
        assert metrics[count_key] == 6
        assert metrics["gate_count"] == 12
        assert metrics["layers"] == 2
        assert metrics[throughput_key] == 24
        assert metrics["process_max_rss_bytes_cumulative"] == 4096
        assert len(fake_cudaq.kernels) == 1


def test_mklq_benchmark_dry_run_accepts_controlled_gate_cases(tmp_path):
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "benchmarks" / "mklq" / "bench_mklq_targets.py"
    output = tmp_path / "dry-run-controlled-rotation.json"
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)

    subprocess.run([
        sys.executable,
        str(script),
        "--dry-run",
        "--targets",
        "mklq-cpu",
        "--cases",
        "ch-state,cy-state,crx-state,cry-state,crz-state",
        "--qubits",
        "4",
        "--output",
        str(output),
    ],
                   check=True,
                   capture_output=True,
                   text=True,
                   env=env)

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["config"]["cases"] == [
        "ch-state",
        "cy-state",
        "crx-state",
        "cry-state",
        "crz-state",
    ]
    rows = report["results"]
    assert len(rows) == 5
    assert {row["status"] for row in rows} == {"planned"}
    assert {row["case"] for row in rows} == {
        "ch-state",
        "cy-state",
        "crx-state",
        "cry-state",
        "crz-state",
    }
    assert {row["estimated_state_bytes"] for row in rows} == {16 * (1 << 4)}


def test_mklq_benchmark_controlled_gate_cases_record_gate_metrics(
        monkeypatch):
    module = _load_benchmark_module()

    class FakeKernel:

        def __init__(self):
            self.operations = []

        def qalloc(self, qubits):
            return list(range(qubits))

        def ry(self, theta, target):
            self.operations.append(("ry", theta, target))

        def rz(self, theta, target):
            self.operations.append(("rz", theta, target))

        def ch(self, control, target):
            self.operations.append(("ch", control, target))

        def cy(self, control, target):
            self.operations.append(("cy", control, target))

        def crx(self, theta, control, target):
            self.operations.append(("crx", theta, control, target))

        def cry(self, theta, control, target):
            self.operations.append(("cry", theta, control, target))

        def crz(self, theta, control, target):
            self.operations.append(("crz", theta, control, target))

    class FakeCudaq:

        def __init__(self):
            self.kernels = []

        def reset_target(self):
            pass

        def set_target(self, target):
            assert target == "mklq-cpu"

        def set_random_seed(self, seed):
            assert seed == 13

        def make_kernel(self):
            kernel = FakeKernel()
            self.kernels.append(kernel)
            return kernel

        def get_state(self, kernel):
            return object()

    def fake_timed_repeats(action, repeats):
        assert repeats == 1
        action()
        return [0.5]

    monkeypatch.setattr(module, "timed_repeats", fake_timed_repeats)
    monkeypatch.setattr(module, "process_max_rss_bytes", lambda: 8192)

    cases = {
        "ch-state": ("ch_gate_count", "ch_gate_state_throughput_per_second"),
        "cy-state": ("cy_gate_count", "cy_gate_state_throughput_per_second"),
        "crx-state": ("crx_gate_count", "crx_gate_state_throughput_per_second"),
        "cry-state": ("cry_gate_count", "cry_gate_state_throughput_per_second"),
        "crz-state": ("crz_gate_count", "crz_gate_state_throughput_per_second"),
    }

    for case, (count_key, throughput_key) in cases.items():
        fake_cudaq = FakeCudaq()
        row = module.run_case(fake_cudaq,
                              "mklq-cpu",
                              case,
                              qubits=4,
                              shots=16,
                              repeats=1,
                              warmups=0,
                              layers=2)

        assert row["status"] == "ok"
        metrics = row["metrics"]
        assert metrics["state_prep_gate_count"] == 8
        assert metrics[count_key] == 6
        assert metrics["gate_count"] == 14
        assert metrics["layers"] == 2
        assert metrics[throughput_key] == 12
        assert metrics["process_max_rss_bytes_cumulative"] == 8192
        assert len(fake_cudaq.kernels) == 1


def test_mklq_benchmark_dry_run_accepts_controlled_case(tmp_path):
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "benchmarks" / "mklq" / "bench_mklq_targets.py"
    output = tmp_path / "dry-run-controlled.json"
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)

    subprocess.run([
        sys.executable,
        str(script),
        "--dry-run",
        "--targets",
        "mklq-cpu",
        "--cases",
        "controlled-state",
        "--qubits",
        "4",
        "--output",
        str(output),
    ],
                   check=True,
                   capture_output=True,
                   text=True,
                   env=env)

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["config"]["cases"] == ["controlled-state"]
    rows = report["results"]
    assert len(rows) == 1
    assert rows[0]["status"] == "planned"
    assert rows[0]["case"] == "controlled-state"
    assert rows[0]["estimated_state_bytes"] == 16 * (1 << 4)


def test_mklq_benchmark_dry_run_accepts_cz_case(tmp_path):
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "benchmarks" / "mklq" / "bench_mklq_targets.py"
    output = tmp_path / "dry-run-cz.json"
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)

    subprocess.run([
        sys.executable,
        str(script),
        "--dry-run",
        "--targets",
        "mklq-cpu",
        "--cases",
        "cz-state",
        "--qubits",
        "4",
        "--output",
        str(output),
    ],
                   check=True,
                   capture_output=True,
                   text=True,
                   env=env)

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["config"]["cases"] == ["cz-state"]
    rows = report["results"]
    assert len(rows) == 1
    assert rows[0]["status"] == "planned"
    assert rows[0]["case"] == "cz-state"
    assert rows[0]["estimated_state_bytes"] == 16 * (1 << 4)


def test_mklq_benchmark_dry_run_accepts_two_qubit_case(tmp_path):
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "benchmarks" / "mklq" / "bench_mklq_targets.py"
    output = tmp_path / "dry-run-two-qubit.json"
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)

    subprocess.run([
        sys.executable,
        str(script),
        "--dry-run",
        "--targets",
        "mklq-cpu",
        "--cases",
        "two-qubit-state",
        "--qubits",
        "4",
        "--output",
        str(output),
    ],
                   check=True,
                   capture_output=True,
                   text=True,
                   env=env)

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["config"]["cases"] == ["two-qubit-state"]
    rows = report["results"]
    assert len(rows) == 1
    assert rows[0]["status"] == "planned"
    assert rows[0]["case"] == "two-qubit-state"
    assert rows[0]["estimated_state_bytes"] == 16 * (1 << 4)


def test_mklq_benchmark_dry_run_accepts_sample_full_register_case(tmp_path):
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "benchmarks" / "mklq" / "bench_mklq_targets.py"
    output = tmp_path / "dry-run-sample-full-register.json"
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)

    subprocess.run([
        sys.executable,
        str(script),
        "--dry-run",
        "--targets",
        "mklq-cpu",
        "--cases",
        "sample-full-register",
        "--qubits",
        "4",
        "--output",
        str(output),
    ],
                   check=True,
                   capture_output=True,
                   text=True,
                   env=env)

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["config"]["cases"] == ["sample-full-register"]
    rows = report["results"]
    assert len(rows) == 1
    assert rows[0]["status"] == "planned"
    assert rows[0]["case"] == "sample-full-register"
    assert rows[0]["estimated_state_bytes"] == 16 * (1 << 4)


def test_mklq_benchmark_dry_run_accepts_sample_basis_case(tmp_path):
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "benchmarks" / "mklq" / "bench_mklq_targets.py"
    output = tmp_path / "dry-run-sample-basis.json"
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)

    subprocess.run([
        sys.executable,
        str(script),
        "--dry-run",
        "--targets",
        "mklq-cpu",
        "--cases",
        "sample-basis",
        "--qubits",
        "4",
        "--output",
        str(output),
    ],
                   check=True,
                   capture_output=True,
                   text=True,
                   env=env)

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["config"]["cases"] == ["sample-basis"]
    rows = report["results"]
    assert len(rows) == 1
    assert rows[0]["status"] == "planned"
    assert rows[0]["case"] == "sample-basis"
    assert rows[0]["estimated_state_bytes"] == 16 * (1 << 4)


def test_mklq_benchmark_dry_run_accepts_sample_partial_register_case(tmp_path):
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "benchmarks" / "mklq" / "bench_mklq_targets.py"
    output = tmp_path / "dry-run-sample-partial-register.json"
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)

    subprocess.run([
        sys.executable,
        str(script),
        "--dry-run",
        "--targets",
        "mklq-metal",
        "--cases",
        "sample-partial-register",
        "--qubits",
        "5",
        "--output",
        str(output),
    ],
                   check=True,
                   capture_output=True,
                   text=True,
                   env=env)

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["config"]["cases"] == ["sample-partial-register"]
    rows = report["results"]
    assert len(rows) == 1
    assert rows[0]["status"] == "planned"
    assert rows[0]["case"] == "sample-partial-register"
    assert rows[0]["estimated_state_bytes"] == 16 * (1 << 5)


def test_mklq_benchmark_dry_run_expands_shot_counts(tmp_path):
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "benchmarks" / "mklq" / "bench_mklq_targets.py"
    output = tmp_path / "dry-run-shot-counts.json"
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)

    subprocess.run([
        sys.executable,
        str(script),
        "--dry-run",
        "--targets",
        "mklq-metal",
        "--cases",
        "sample-full-register,sample-partial-register",
        "--qubits",
        "5",
        "--shot-counts",
        "16,64,256",
        "--output",
        str(output),
    ],
                   check=True,
                   capture_output=True,
                   text=True,
                   env=env)

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["config"]["shot_counts"] == [16, 64, 256]
    rows = report["results"]
    assert len(rows) == 6
    assert {row["shots"] for row in rows} == {16, 64, 256}
    assert {row["case"] for row in rows} == {
        "sample-full-register",
        "sample-partial-register",
    }


def test_mklq_benchmark_rejects_invalid_shot_counts(tmp_path):
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "benchmarks" / "mklq" / "bench_mklq_targets.py"
    output = tmp_path / "invalid-shot-counts.json"
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)

    result = subprocess.run([
        sys.executable,
        str(script),
        "--dry-run",
        "--targets",
        "mklq-metal",
        "--cases",
        "sample-full-register",
        "--qubits",
        "5",
        "--shot-counts",
        "0,16",
        "--output",
        str(output),
    ],
                            capture_output=True,
                            text=True,
                            env=env)

    assert result.returncode != 0
    assert "expected positive integer" in result.stderr
    assert not output.exists()


def test_mklq_probability_microbenchmark_dry_run_writes_schema(tmp_path):
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "benchmarks" / "mklq" / "bench_probability_kernels.py"
    output = tmp_path / "probability-dry-run.json"

    subprocess.run([
        sys.executable,
        str(script),
        "--dry-run",
        "--variants",
        "scalar-norm,scalar-split",
        "--qubits",
        "4,5",
        "--repeats",
        "2",
        "--output",
        str(output),
    ],
                   check=True,
                   capture_output=True,
                   text=True)

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["schema_version"] == "mklq-probability-benchmark-v1"
    assert report["config"]["variants"] == ["scalar-norm", "scalar-split"]
    assert report["config"]["qubits"] == [4, 5]
    assert report["config"]["dry_run"] is True
    rows = report["results"]
    assert len(rows) == 4
    assert {row["status"] for row in rows} == {"planned"}
    assert {row["variant"] for row in rows} == {"scalar-norm", "scalar-split"}
    assert {row["qubits"] for row in rows} == {4, 5}


def test_mklq_probability_microbenchmark_defaults_cover_runtime_vdsp_path():
    module = _load_probability_benchmark_module()

    assert "accelerate-interleaved" in module.DEFAULT_VARIANTS


def test_mklq_probability_microbenchmark_records_non_dry_run_schema(monkeypatch):
    module = _load_probability_benchmark_module()
    binary = Path("/tmp/fake-mklq-probability-binary")
    compile_metadata = {
        "compiler": "/usr/bin/clang++",
        "command": ["/usr/bin/clang++", "probability_kernels.cpp"],
        "returncode": 0,
        "stdout": "",
        "stderr": "",
        "openmp_enabled": True,
        "accelerate_enabled": True,
        "binary": str(binary),
    }

    def fake_compile(args):
        assert args.variants == ["openmp-split"]
        return binary, compile_metadata

    def fake_run(command, capture_output=False, text=False, **kwargs):
        if command[0] != str(binary):
            return subprocess.CompletedProcess(command,
                                               returncode=0,
                                               stdout="",
                                               stderr="")
        payload = {
            "results": [{
                "variant": "openmp-split",
                "qubits": 4,
                "dimension": 16,
                "status": "ok",
                "metrics": {
                    "elapsed_seconds_min": 1.0e-6,
                    "elapsed_seconds_median": 2.0e-6,
                    "elapsed_seconds_max": 3.0e-6,
                    "state_amplitudes_per_second": 8.0e6,
                    "max_abs_diff_vs_scalar_norm": 0.0,
                    "probability_checksum": 1.0,
                    "openmp_threads": 4,
                },
            }]
        }
        return subprocess.CompletedProcess(command,
                                           returncode=0,
                                           stdout=json.dumps(payload),
                                           stderr="")

    monkeypatch.setattr(module, "compile_binary", fake_compile)
    monkeypatch.setattr(module.subprocess, "run", fake_run)
    args = argparse.Namespace(variants=["openmp-split"],
                              qubits=[4],
                              repeats=2,
                              warmups=1,
                              seed=13,
                              dry_run=False,
                              binary=None)

    report = module.build_report(args)

    assert report["schema_version"] == "mklq-probability-benchmark-v1"
    assert report["config"]["dry_run"] is False
    assert report["compile"] == compile_metadata
    assert report["execution"]["returncode"] == 0
    assert report["execution"]["command"][0] == str(binary)
    row = report["results"][0]
    assert row["status"] == "ok"
    assert row["repeats"] == 2
    assert row["metrics"]["openmp_threads"] == 4
    assert row["metrics"]["max_abs_diff_vs_scalar_norm"] == 0.0


def test_mklq_benchmark_isolated_row_error_is_reported(tmp_path):
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "benchmarks" / "mklq" / "bench_mklq_targets.py"
    output = tmp_path / "isolated-error.json"

    result = subprocess.run([
        sys.executable,
        str(script),
        "--isolate-rows",
        "--targets",
        "not-a-target",
        "--cases",
        "gate-state",
        "--qubits",
        "1",
        "--repeats",
        "1",
        "--warmups",
        "1",
        "--layers",
        "1",
        "--output",
        str(output),
    ],
                            capture_output=True,
                            text=True)

    assert result.returncode == 1
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["config"]["isolate_rows"] is True
    row = report["results"][0]
    assert row["status"] == "error"
    assert "not-a-target" in row["error"]
    assert row["isolated_process"]["returncode"] == 0
    assert "stdout" in row["isolated_process"]
    assert "stderr" in row["isolated_process"]
    assert row["isolated_process"]["runtime"]["cudaq_module_file"]


def test_mklq_benchmark_isolated_malformed_json_returns_error(monkeypatch):
    module = _load_benchmark_module()

    def fake_run(command, capture_output, text):
        output = Path(command[command.index("--output") + 1])
        output.write_text("{not-json", encoding="utf-8")
        return subprocess.CompletedProcess(command,
                                           returncode=0,
                                           stdout="child stdout",
                                           stderr="child stderr")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    args = argparse.Namespace(shots=8, repeats=1, warmups=1, layers=1)

    row = module.run_isolated_case(args, "mklq-cpu", "gate-state", 1)

    assert row["status"] == "error"
    assert "invalid isolated benchmark JSON" in row["error"]
    assert row["isolated_process"]["returncode"] == 0
    assert row["isolated_process"]["stdout"] == "child stdout"
    assert row["isolated_process"]["stderr"] == "child stderr"


def test_mklq_benchmark_summary_records_clean_cpu_evidence():
    repo_root = Path(__file__).resolve().parents[3]
    summary_path = (
        repo_root / "benchmarks" / "mklq" / "reports" /
        "local-clean-cpu-q20-2026-06-21.summary.json")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert summary["schema_version"] == "mklq-benchmark-summary-v1"
    assert summary["evidence_kind"] == "clean_local_benchmark_evidence"
    assert summary["summary_id"] == "local-clean-cpu-q20-2026-06-21"
    assert summary["git"]["commit"] == (
        "4b112725f557f537d314d7554879cca39d6b14d5")
    assert summary["git"]["dirty"] is False
    assert summary["interpretation"]["clean_worktree"] is True
    assert summary["raw_results"][0]["sha256"] == (
        "ebc3d0671c1c3009fce77c2b6d54e25e7589778e0adf40a2cc83f2f546aa7ce9")
    assert summary["raw_results"][0]["status_rows"] == {"ok": 4}
    assert summary["raw_results"][1]["sha256"] == (
        "8bac8a02ec19503c569ac2e953fe633e85aa43e78a80418ff8a1b6334772ca1e")
    assert summary["raw_results"][1]["status_rows"] == {"ok": 8}
    assert summary["machine"]["cpu_brand"] == "Apple M5"
    assert summary["config"]["targets"] == ["qpp-cpu", "mklq-cpu"]
    assert summary["config"]["cases"] == [
        "y-state", "cy-state", "sample-full-register",
        "sample-partial-register"
    ]
    assert summary["config"]["qubits"] == [20]
    assert summary["config"]["shot_counts"] == [1024, 65536]
    assert summary["config"]["repeats"] == 2
    assert summary["config"]["warmups"] == 1
    assert summary["config"]["layers"] == 8

    rows = {
        (row["target"], row["case"], row["shots"]): row
        for row in summary["rows"]
    }
    assert len(rows) == 12
    assert rows[("mklq-cpu", "y-state", 1024)][
        "elapsed_seconds_median"] == 0.053900624508969486
    assert rows[("mklq-cpu", "sample-partial-register", 65536)][
        "elapsed_seconds_median"] == 0.011505229005706497
    ratios = summary["comparison"]["clean_worktree_cross_target_ratio"]
    assert ratios["qpp_cpu_over_mklq_cpu_y_state_q20"] == pytest.approx(
        96.80846823274014)
    assert ratios[
        "qpp_cpu_over_mklq_cpu_sample_partial_register_q20_65536_shots"
    ] == pytest.approx(99.64972043797643)


def test_mklq_benchmark_summary_records_sanitized_sampling_evidence():
    repo_root = Path(__file__).resolve().parents[3]
    summary_path = (
        repo_root / "benchmarks" / "mklq" / "reports" /
        "local-current-sampling-fullprob-gated-q20-2026-06-19.summary.json")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert summary["schema_version"] == "mklq-benchmark-summary-v1"
    assert summary["evidence_kind"] == "local_tuning_evidence"
    assert summary["raw_results"][0]["path"] == (
        "benchmarks/mklq/results/"
        "local-current-sampling-fullprob-gated-q20-2026-06-19.json")
    assert summary["raw_results"][0]["sha256"] == (
        "8ca6a4f7a7aea1670aa572ea6897a125ea4ff0a9e0d1d93502c1158e81ba33b3")
    assert summary["raw_results"][1]["sha256"] == (
        "9c15c0c1d566f0270294b157b7ef2d6834bedf421009e10263903547496f10b1")
    assert summary["machine"]["cpu_brand"] == "Apple M5"
    assert summary["machine"]["logical_cores"] == 10
    assert summary["machine"]["memory_bytes"] == 17179869184
    assert summary["machine"]["macos_version"] == "26.5.1"
    assert summary["config"]["targets"] == ["qpp-cpu", "mklq-cpu",
                                             "mklq-metal"]
    assert summary["config"]["cases"] == [
        "sample-full-register", "sample-partial-register"
    ]
    assert summary["config"]["qubits"] == [20]
    assert summary["config"]["shots"] == 1024
    assert summary["config"]["repeats"] == 2
    assert summary["config"]["warmups"] == 1
    assert summary["config"]["layers"] == 4

    rows = {
        (row["target"], row["case"]): row
        for row in summary["rows"]
    }
    assert rows[("mklq-metal", "sample-partial-register")][
        "elapsed_seconds_median"] == 0.022011521003150847
    assert rows[("mklq-metal", "sample-partial-register")][
        "sample_path"] == "resident_full_register_probability_fill_host_fold"
    assert rows[("mklq-metal", "sample-full-register")][
        "elapsed_seconds_median"] == 0.03705766650091391
    assert summary["comparison"]["pre_gate_probe"][
        "mklq_metal_sample_partial_register_q20_seconds"] == 0.2556968749995576


def test_mklq_benchmark_summary_records_counts_only_sampling_shot_scaling():
    repo_root = Path(__file__).resolve().parents[3]
    summary_path = (
        repo_root / "benchmarks" / "mklq" / "reports" /
        "local-counts-only-sampling-shot-scaling-q20-2026-06-19.summary.json")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert summary["schema_version"] == "mklq-benchmark-summary-v1"
    assert summary["evidence_kind"] == "local_tuning_evidence"
    assert summary["summary_id"] == (
        "local-counts-only-sampling-shot-scaling-q20-2026-06-19")
    assert summary["raw_results"][0]["path"] == (
        "benchmarks/mklq/results/"
        "local-counts-only-sampling-shot-scaling-q20-2026-06-19.json")
    assert summary["raw_results"][0]["sha256"] == (
        "ef9846673b461e3abc6d359933408be58e1f745d8b68738b757a76339f9b5092")
    assert summary["raw_results"][0]["status_rows"] == {"ok": 24}
    assert summary["machine"]["cpu_brand"] == "Apple M5"
    assert summary["config"]["targets"] == ["qpp-cpu", "mklq-cpu",
                                             "mklq-metal"]
    assert summary["config"]["cases"] == [
        "sample-full-register", "sample-partial-register"
    ]
    assert summary["config"]["qubits"] == [20]
    assert summary["config"]["shot_counts"] == [256, 1024, 8192, 65536]
    assert summary["config"]["repeats"] == 2
    assert summary["config"]["warmups"] == 1
    assert summary["config"]["layers"] == 8

    rows = {
        (row["target"], row["case"], row["shots"]): row
        for row in summary["rows"]
    }
    assert len(rows) == 24
    assert rows[("mklq-cpu", "sample-full-register", 65536)][
        "sample_path"] == "mklq_counts_only_backend_sample"
    assert rows[("mklq-cpu", "sample-partial-register", 65536)][
        "sample_path"] == "mklq_counts_only_backend_sample"
    assert rows[("mklq-metal", "sample-full-register", 65536)][
        "sample_path"] == "mklq_metal_mixed_path_host_counts"
    assert rows[("mklq-metal", "sample-partial-register", 65536)][
        "sample_path"] == "mklq_metal_mixed_path_host_counts"
    assert summary["interpretation"]["standard_sample_counts_only_path"]
    assert summary["interpretation"]["do_not_treat_as_clean_release_provenance"]


def test_mklq_benchmark_summary_records_y_cy_fastpath_evidence():
    repo_root = Path(__file__).resolve().parents[3]
    summary_path = (
        repo_root / "benchmarks" / "mklq" / "reports" /
        "local-y-cy-fastpath-isolated-q20-2026-06-19.summary.json")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert summary["schema_version"] == "mklq-benchmark-summary-v1"
    assert summary["evidence_kind"] == "local_tuning_evidence"
    assert summary["raw_results"][0]["path"] == (
        "benchmarks/mklq/results/"
        "local-y-cy-fastpath-isolated-q20-2026-06-19.json")
    assert summary["raw_results"][0]["sha256"] == (
        "93bce3b77fccce0ce48611fbccc2a88d81e31b8a34f4885ff9235750178701fa")
    assert summary["machine"]["cpu_brand"] == "Apple M5"
    assert summary["config"]["targets"] == ["qpp-cpu", "mklq-cpu"]
    assert summary["config"]["cases"] == ["y-state", "cy-state"]
    assert summary["config"]["qubits"] == [20]

    rows = {
        (row["target"], row["case"]): row
        for row in summary["rows"]
    }
    assert rows[("mklq-cpu", "y-state")][
        "y_gate_state_throughput_per_second"] == 3322.8671668028323
    assert rows[("mklq-cpu", "cy-state")][
        "cy_gate_state_throughput_per_second"] == 1765.9796294202324
    assert summary["comparison"]["same_day_cross_target_ratio"][
        "qpp_cpu_over_mklq_cpu_y_state_q20"] == 167.37794366574514
    assert summary["comparison"]["same_day_cross_target_ratio"][
        "qpp_cpu_over_mklq_cpu_cy_state_q20"] == 103.84948452737598
    assert summary["interpretation"]["do_not_treat_as_clean_release_provenance"]


def test_mklq_benchmark_summary_records_metal_y_cy_resident_evidence():
    repo_root = Path(__file__).resolve().parents[3]
    summary_path = (
        repo_root / "benchmarks" / "mklq" / "reports" /
        "local-metal-y-cy-resident-isolated-q20-2026-06-19.summary.json")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert summary["schema_version"] == "mklq-benchmark-summary-v1"
    assert summary["evidence_kind"] == "local_tuning_evidence"
    assert summary["raw_results"][0]["path"] == (
        "benchmarks/mklq/results/"
        "local-metal-y-cy-resident-isolated-q20-2026-06-19.json")
    assert summary["raw_results"][0]["sha256"] == (
        "84891e8f907c38295a4975b1d0b0c493c2658b9b36b29975c539b93fcdfff9bb")
    assert summary["machine"]["cpu_brand"] == "Apple M5"
    assert summary["config"]["targets"] == ["qpp-cpu", "mklq-cpu",
                                             "mklq-metal"]
    assert summary["config"]["cases"] == ["y-state", "cy-state"]
    assert summary["config"]["qubits"] == [20]
    assert summary["config"]["layers"] == 8

    rows = {
        (row["target"], row["case"]): row
        for row in summary["rows"]
    }
    assert rows[("mklq-metal", "y-state")][
        "curated_path_label"] == "mklq_metal_resident_y_gate_path"
    assert rows[("mklq-metal", "cy-state")][
        "curated_path_label"] == (
            "mklq_metal_resident_controlled_y_gate_path")
    assert rows[("mklq-metal", "y-state")]["path_label_source"] == (
        "inferred_from_runtime_tests_and_code_inspection")
    assert rows[("mklq-metal", "cy-state")]["path_label_source"] == (
        "inferred_from_runtime_tests_and_code_inspection")
    assert rows[("mklq-metal", "y-state")][
        "y_gate_state_throughput_per_second"] > 0.0
    assert rows[("mklq-metal", "cy-state")][
        "cy_gate_state_throughput_per_second"] > 0.0
    assert summary["interpretation"]["metal_path_scope"] == (
        "resident fp32 Metal gate update followed by host readback for "
        "cudaq.get_state")
    assert summary["interpretation"]["do_not_treat_as_clean_release_provenance"]
