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


def _load_public_healthcheck_module():
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "benchmarks" / "mklq" / "run_public_healthcheck.py"
    spec = importlib.util.spec_from_file_location("run_public_healthcheck",
                                                  script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_public_readiness_audit_module():
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "benchmarks" / "mklq" / (
        "run_public_readiness_audit.py")
    spec = importlib.util.spec_from_file_location("run_public_readiness_audit",
                                                  script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_preflight_audit_module():
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "benchmarks" / "mklq" / "run_preflight_audit.py"
    spec = importlib.util.spec_from_file_location("run_preflight_audit",
                                                  script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_example_verifier_module():
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "examples" / "mklq" / "verify_examples.py"
    spec = importlib.util.spec_from_file_location("verify_examples", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_performance_evidence_module():
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "benchmarks" / "mklq" / (
        "check_performance_evidence.py")
    spec = importlib.util.spec_from_file_location("check_performance_evidence",
                                                  script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_metal_evidence_module():
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "benchmarks" / "mklq" / "check_metal_evidence.py"
    spec = importlib.util.spec_from_file_location("check_metal_evidence",
                                                  script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_metal_runtime_counter_probe_module():
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "benchmarks" / "mklq" / (
        "run_metal_runtime_counter_probe.py")
    spec = importlib.util.spec_from_file_location(
        "run_metal_runtime_counter_probe", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_metal_runtime_counter_summary_module():
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "benchmarks" / "mklq" / (
        "summarize_metal_runtime_counters.py")
    spec = importlib.util.spec_from_file_location(
        "summarize_metal_runtime_counters", script)
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


def test_mklq_summary_generator_accepts_extra_interpretation_fields(tmp_path):
    module = _load_summary_generator_module()
    raw_path = tmp_path / "metal.json"
    metal_row = _benchmark_row("mklq-metal", "qft-like-state", 2.0)
    metal_row["metrics"].update({
        "metal_path_label": "mklq_metal_mixed_composite_state_host_readback",
        "metal_path_label_source": "benchmark_harness_static_case_map",
        "metal_full_native": False,
        "metal_runtime_counter": False,
    })
    raw_path.write_text(json.dumps(
        _raw_benchmark_report(cases=["qft-like-state"],
                              results=[
                                  _benchmark_row("qpp-cpu", "qft-like-state",
                                                 10.0),
                                  metal_row,
                              ])),
                        encoding="utf-8")

    summary = module.build_summary(
        raw_paths=[raw_path],
        summary_id="local-metal-composite-test",
        evidence_kind="local_tuning_evidence",
        reference_target="qpp-cpu",
        candidate_target="mklq-metal",
        ratio_group="same_day_cross_target_ratio",
        performance_scope="local tuning evidence only",
        summary_text="Synthetic Metal composite summary.",
        extra_interpretation={
            "do_not_treat_as_clean_release_provenance": True,
            "metal_path_scope": (
                "mixed-path Metal state update followed by host readback"),
            "notes": ["generated", "bounded"],
        },
    )

    assert summary["interpretation"][
        "do_not_treat_as_clean_release_provenance"] is True
    assert summary["interpretation"]["metal_path_scope"] == (
        "mixed-path Metal state update followed by host readback")
    assert summary["interpretation"]["notes"] == ["generated", "bounded"]
    ratios = summary["comparison"]["same_day_cross_target_ratio"]
    assert ratios["qpp_cpu_over_mklq_metal_qft_like_state_q20"] == 5.0
    elapsed = summary["comparison"]["mklq_metal_elapsed_seconds_median"]
    assert elapsed["qft_like_state_q20"] == 2.0
    metal_summary_row = [
        row for row in summary["rows"] if row["target"] == "mklq-metal"
    ][0]
    assert metal_summary_row["metal_path_label"] == (
        "mklq_metal_mixed_composite_state_host_readback")
    assert metal_summary_row["metal_path_label_source"] == (
        "benchmark_harness_static_case_map")
    assert metal_summary_row["metal_full_native"] is False
    assert metal_summary_row["metal_runtime_counter"] is False


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
        gate_cases="y-state,cy-state,cz-state",
        composite_cases="qft-like-state,seeded-clifford-state",
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
        "local-clean-cpu-gate-y-cy-cz-q20-2026-06-21.json")
    assert plan["paths"]["composite_raw"].endswith(
        "local-clean-cpu-composite-qft-like-seeded-clifford-q20-2026-06-21.json"
    )
    assert plan["paths"]["sampling_raw"].endswith(
        "local-clean-cpu-sampling-q20-2026-06-21.json")
    assert plan["paths"]["summary"].endswith(
        "local-clean-cpu-q20-2026-06-21.summary.json")
    gate_command = plan["commands"]["gate_raw"]
    assert "--isolate-rows" in gate_command
    assert gate_command[gate_command.index("--cases") + 1] == (
        "y-state,cy-state,cz-state")
    assert gate_command[gate_command.index("--targets") + 1] == (
        "qpp-cpu,mklq-cpu")
    composite_command = plan["commands"]["composite_raw"]
    assert composite_command[composite_command.index("--cases") + 1] == (
        "qft-like-state,seeded-clifford-state")
    assert composite_command[composite_command.index("--layers") + 1] == "8"
    sampling_command = plan["commands"]["sampling_raw"]
    assert sampling_command[sampling_command.index("--shot-counts") +
                            1] == "1024,65536"
    summary_command = plan["commands"]["summary"]
    assert summary_command.count("--raw") == 3
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
        composite_cases="qft-like-state,seeded-clifford-state",
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
        skip_metal_counter_probe=False,
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
        "metal_runtime_counter_probe",
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
    assert plan["steps"][3]["command"][:2] == [
        "/usr/bin/python3",
        "benchmarks/mklq/run_metal_runtime_counter_probe.py",
    ]
    assert "--output" in plan["steps"][3]["command"]
    assert (plan["steps"][3]["command"][plan["steps"][3]["command"].index(
        "--output") + 1] ==
            "benchmarks/mklq/results/local-metal-runtime-counter-probe-2026-06-21.counter.json"
            )


def test_mklq_correctness_gate_can_skip_metal_counter_probe(tmp_path):
    module = _load_correctness_gate_module()
    base = _correctness_gate_config(module, tmp_path)
    config = module.CorrectnessGateConfig(**{
        **base.__dict__,
        "skip_metal_counter_probe": True,
    })

    plan = module.build_plan(config)

    assert [step["name"] for step in plan["steps"]] == [
        "python_target_smoke",
        "nvqpp_smoke",
        "target_config_ctest",
    ]
    assert plan["skipped_steps"] == ["metal_runtime_counter_probe"]


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
        "passed": 4,
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


def _public_healthcheck_config(module,
                               tmp_path,
                               *,
                               full=False,
                               include_harness_tests=True,
                               refresh_clean_cpu_benchmark=False,
                               plan_only=False):
    return module.HealthcheckConfig(
        repo_root=tmp_path,
        install_prefix=tmp_path / "install",
        build_dir=tmp_path / "build-python",
        python_executable=sys.executable,
        pythonpath=str(tmp_path / "install"),
        nvqpp=tmp_path / "install" / "bin" / "nvq++",
        stamp="2026-06-21",
        output=tmp_path / "results" / "public-healthcheck.json",
        jobs=3,
        timeout_seconds=123,
        tail_chars=80,
        require_clean=False,
        full=full,
        include_harness_tests=include_harness_tests,
        refresh_clean_cpu_benchmark=refresh_clean_cpu_benchmark,
        plan_only=plan_only,
    )


def test_mklq_public_healthcheck_default_output_uses_mode_label():
    module = _load_public_healthcheck_module()

    assert module.output_default("2026-06-21") == Path(
        "benchmarks/mklq/results/public-healthcheck-2026-06-21.json")
    assert module.output_default("2026-06-21", full=True) == Path(
        "benchmarks/mklq/results/public-healthcheck-full-2026-06-21.json")
    assert module.output_default(
        "2026-06-21", refresh_clean_cpu_benchmark=True) == Path(
            "benchmarks/mklq/results/"
            "public-healthcheck-refresh-clean-cpu-2026-06-21.json")
    assert module.output_default(
        "2026-06-21", full=True, refresh_clean_cpu_benchmark=True) == Path(
            "benchmarks/mklq/results/"
            "public-healthcheck-full-refresh-clean-cpu-2026-06-21.json")


def test_mklq_public_healthcheck_plan_lists_escalating_gates(tmp_path):
    module = _load_public_healthcheck_module()
    config = _public_healthcheck_config(module,
                                        tmp_path,
                                        full=True,
                                        refresh_clean_cpu_benchmark=True,
                                        plan_only=True)

    report = module.run_healthcheck(config)

    assert report["schema_version"] == module.SCHEMA_VERSION
    assert report["summary"]["status"] == "planned"
    assert [step["name"] for step in report["steps"]] == [
        "git_repository",
        "tracked_artifacts",
        "public_metadata",
        "benchmark_summary_parse",
        "performance_evidence_guard",
        "metal_evidence_guard",
        "metal_runtime_counter_probe_parse",
        "benchmark_helper_py_compile",
        "example_source_files",
        "markdown_links",
        "benchmark_evidence_regeneration",
        "benchmark_harness_tests",
        "install_prefix_build",
        "correctness_gate",
        "example_smoke",
        "clean_cpu_benchmark",
    ]
    assert not config.output.exists()


def test_mklq_public_healthcheck_rejects_tracked_artifacts(monkeypatch,
                                                          tmp_path):
    module = _load_public_healthcheck_module()
    config = _public_healthcheck_config(module, tmp_path)

    monkeypatch.setattr(
        module, "command_output",
        lambda root, args: "README.md\nbuild/generated.o\nbenchmarks/mklq/results/raw.json"
    )

    result = module.run_tracked_artifact_check(config)

    assert result["status"] == "failed"
    assert result["details"]["bad_paths"] == [
        "build/generated.o", "benchmarks/mklq/results/raw.json"
    ]


def test_mklq_public_healthcheck_can_require_clean_worktree(monkeypatch,
                                                            tmp_path):
    module = _load_public_healthcheck_module()
    base = _public_healthcheck_config(module, tmp_path)
    config = module.HealthcheckConfig(**{
        **base.__dict__,
        "require_clean": True,
    })

    def fake_command_output(root, args):
        if args == ["git", "status", "--short", "--branch"]:
            return "## main\n M README.md\n?? scratch.txt"
        if args == ["git", "remote", "-v"]:
            return "\n".join([
                "origin\thttps://github.com/wuls968/MKL-Q.git (fetch)",
                "origin\thttps://github.com/wuls968/MKL-Q.git (push)",
                "upstream\thttps://github.com/NVIDIA/cuda-quantum.git (fetch)",
            ])
        if args == ["git", "rev-parse", "--is-shallow-repository"]:
            return "false"
        if args == ["git", "ls-files", ".github/workflows"]:
            return ".github/workflows/mklq-public-hygiene.yml"
        raise AssertionError(args)

    monkeypatch.setattr(module, "command_output", fake_command_output)

    result = module.run_git_repository_check(config)

    assert result["status"] == "failed"
    assert "working tree is dirty" in result["message"]


def test_mklq_public_healthcheck_checks_metadata_tokens(monkeypatch, tmp_path):
    module = _load_public_healthcheck_module()
    config = _public_healthcheck_config(module, tmp_path)
    readme = tmp_path / "README.md"
    readme.write_text("MKL-Q source-only mklq-cpu", encoding="utf-8")

    monkeypatch.setattr(module, "public_metadata_requirements",
                        lambda: [("README.md", "MKL-Q")])
    monkeypatch.setattr(module, "public_metadata_paths", lambda root: [readme])

    assert module.run_public_metadata_check(config)["status"] == "passed"

    readme.write_text(module.banned_tokens()[0], encoding="utf-8")

    result = module.run_public_metadata_check(config)
    assert result["status"] == "failed"
    assert "banned_token_failures" in result["details"]


def test_mklq_public_healthcheck_requires_metal_guard_in_pr_template():
    module = _load_public_healthcheck_module()

    requirements = set(module.public_metadata_requirements())

    assert (".github/pull_request_template.md",
            "check_metal_evidence.py") in requirements


def test_mklq_public_healthcheck_checks_example_sources(tmp_path):
    module = _load_public_healthcheck_module()
    config = _public_healthcheck_config(module, tmp_path)
    assert "examples/mklq/python/parametric.py" in module.EXAMPLE_SOURCE_FILES
    assert "examples/mklq/cpp/phase_kickback.cpp" in module.EXAMPLE_SOURCE_FILES
    assert "examples/mklq/cpp/clifford_chain.cpp" in module.EXAMPLE_SOURCE_FILES
    for relative_path in module.EXAMPLE_SOURCE_FILES:
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("example\n", encoding="utf-8")

    assert module.run_example_source_check(config)["status"] == "passed"

    (tmp_path / "examples" / "mklq" / "cpp" / "ghz.cpp").unlink()

    result = module.run_example_source_check(config)
    assert result["status"] == "failed"
    assert "examples/mklq/cpp/ghz.cpp" in result["details"]["missing"]


def test_mklq_example_verifier_plans_and_checks_bitstrings(tmp_path):
    module = _load_example_verifier_module()
    assert module.DEFAULT_EXAMPLES == (
        "bell",
        "ghz",
        "parametric",
        "phase_kickback",
        "clifford_chain",
    )
    config = module.ExampleConfig(
        repo_root=tmp_path,
        install_prefix=tmp_path / "install",
        pythonpath=str(tmp_path / "install"),
        nvqpp=tmp_path / "install" / "bin" / "nvq++",
        python_executable=sys.executable,
        targets=["mklq-cpu"],
        examples=["bell"],
        shots=5,
        output=tmp_path / "example-smoke.json",
        timeout_seconds=10,
        plan_only=True,
        skip_python=False,
        skip_cpp=False,
    )

    report = module.run_examples(config)

    assert report["schema_version"] == module.SCHEMA_VERSION
    assert report["summary"] == {
        "status": "planned",
        "planned": 3,
        "failed": 0,
    }
    assert [step["name"] for step in report["steps"]] == [
        "python_bell_mklq_cpu",
        "cpp_compile_bell_mklq_cpu",
        "cpp_run_bell_mklq_cpu",
    ]
    assert module.validate_counts("bell", "mklq-cpu: { 00:3 11:2 }")[
        "counts_ok"] is True
    bad = module.validate_counts("bell", "{ 00:3 01:2 }")
    assert bad["counts_ok"] is False
    assert bad["unexpected_bitstrings"] == ["01"]
    assert module.validate_counts("parametric", "{ 111:5 }")[
        "counts_ok"] is True
    assert module.validate_counts("phase_kickback", "{ 11:5 }")[
        "counts_ok"] is True
    assert module.validate_counts("clifford_chain", "{ 1111:5 }")[
        "counts_ok"] is True
    clifford_bad = module.validate_counts("clifford_chain", "{ 1011:5 }")
    assert clifford_bad["counts_ok"] is False
    assert clifford_bad["unexpected_bitstrings"] == ["1011"]


def test_mklq_example_verifier_default_plan_covers_public_fixtures(tmp_path):
    module = _load_example_verifier_module()
    config = module.ExampleConfig(
        repo_root=tmp_path,
        install_prefix=tmp_path / "install",
        pythonpath=str(tmp_path / "install"),
        nvqpp=tmp_path / "install" / "bin" / "nvq++",
        python_executable=sys.executable,
        targets=list(module.DEFAULT_TARGETS),
        examples=list(module.DEFAULT_EXAMPLES),
        shots=5,
        output=tmp_path / "example-smoke.json",
        timeout_seconds=10,
        plan_only=True,
        skip_python=False,
        skip_cpp=False,
    )

    report = module.run_examples(config)

    assert report["summary"] == {
        "status": "planned",
        "planned": 30,
        "failed": 0,
    }
    planned_names = {step["name"] for step in report["steps"]}
    assert "python_parametric_mklq_cpu" in planned_names
    assert "cpp_run_phase_kickback_mklq_metal" in planned_names
    assert "cpp_compile_clifford_chain_mklq_cpu" in planned_names


def test_mklq_public_healthcheck_invokes_example_verifier(monkeypatch,
                                                          tmp_path):
    module = _load_public_healthcheck_module()
    config = _public_healthcheck_config(module, tmp_path, full=True)
    calls = []

    def fake_run_command(config, command, env_overlay=None):
        calls.append(command)
        return {
            "returncode": 0,
            "command": command,
            "stdout_tail": "{}",
            "stderr_tail": "",
        }

    monkeypatch.setattr(module, "run_command", fake_run_command)

    result = module.run_example_smoke(config)

    assert result["status"] == "passed"
    command = calls[0]
    assert command[1].endswith("examples/mklq/verify_examples.py")
    assert "--install-prefix" in command
    assert "--pythonpath" in command
    assert "--nvqpp" in command
    assert "--shots" in command


def test_mklq_public_healthcheck_compares_benchmark_evidence(monkeypatch,
                                                             tmp_path):
    module = _load_public_healthcheck_module()
    config = _public_healthcheck_config(module, tmp_path)
    expected = tmp_path / "docs" / "mklq" / "benchmark-evidence.md"
    expected.parent.mkdir(parents=True)
    expected.write_text("expected evidence\n", encoding="utf-8")

    def fake_run_command(config, command, env_overlay=None):
        Path(command[-1]).write_text("expected evidence\n", encoding="utf-8")
        return {"returncode": 0, "command": command}

    monkeypatch.setattr(module, "run_command", fake_run_command)

    assert module.run_benchmark_evidence_check(config)["status"] == "passed"

    def fake_stale_run_command(config, command, env_overlay=None):
        Path(command[-1]).write_text("stale evidence\n", encoding="utf-8")
        return {"returncode": 0, "command": command}

    monkeypatch.setattr(module, "run_command", fake_stale_run_command)

    result = module.run_benchmark_evidence_check(config)
    assert result["status"] == "failed"
    assert "differs" in result["message"]


def _performance_summary(module,
                         *,
                         ratio=20.0,
                         dirty=False,
                         performance_scope=None,
                         missing_label=None):
    required = list(module.DEFAULT_REQUIRED_RATIOS)
    ratios = {
        module.ratio_key("qpp-cpu", "mklq-cpu", label): ratio
        for label in required
    }
    elapsed = {label: 0.01 for label in required}
    if missing_label:
        ratios.pop(module.ratio_key("qpp-cpu", "mklq-cpu", missing_label))
    return {
        "schema_version": module.SUMMARY_SCHEMA_VERSION,
        "evidence_kind": module.DEFAULT_EVIDENCE_KIND,
        "summary_id": "local-clean-cpu-test",
        "git": {
            "commit": "abc123",
            "dirty": dirty,
        },
        "config": {
            "targets": ["qpp-cpu", "mklq-cpu"],
            "qubits": [20],
            "repeats": 2,
            "warmups": 1,
            "isolate_rows": True,
        },
        "raw_results": [{
            "path": "benchmarks/mklq/results/local-clean.json",
            "sha256": "a" * 64,
            "status_rows": {
                "ok": 12,
            },
            "tracked": False,
        }],
        "rows": [{
            "target": "mklq-cpu",
            "case": "y-state",
            "qubits": 20,
            "shots": 1024,
            "status": "ok",
        }],
        "comparison": {
            module.DEFAULT_RATIO_GROUP: ratios,
            module.DEFAULT_CANDIDATE_ELAPSED_GROUP: elapsed,
        },
        "interpretation": {
            "clean_worktree": not dirty,
            "raw_json_files_are_ignored": True,
            "performance_claim_scope": performance_scope or (
                "local evidence only; not cross-machine performance "
                "certification"),
        },
    }


def test_mklq_performance_evidence_guard_accepts_clean_cpu_summary():
    module = _load_performance_evidence_module()
    assert "qft_like_state_q20" in module.DEFAULT_REQUIRED_RATIOS
    assert "seeded_clifford_state_q20" in module.DEFAULT_REQUIRED_RATIOS

    result = module.check_summary(
        _performance_summary(module),
        required_ratios=list(module.DEFAULT_REQUIRED_RATIOS),
        min_speedup=10.0,
        ratio_group=module.DEFAULT_RATIO_GROUP,
        candidate_elapsed_group=module.DEFAULT_CANDIDATE_ELAPSED_GROUP,
        reference_target="qpp-cpu",
        candidate_target="mklq-cpu",
    )

    assert result["status"] == "passed"
    assert result["checked_ratio_count"] == len(module.DEFAULT_REQUIRED_RATIOS)
    assert result["min_checked_speedup"] == 20.0


def test_mklq_performance_evidence_guard_rejects_weak_or_dirty_summary():
    module = _load_performance_evidence_module()

    weak = module.check_summary(
        _performance_summary(module, ratio=2.0),
        required_ratios=list(module.DEFAULT_REQUIRED_RATIOS),
        min_speedup=10.0,
        ratio_group=module.DEFAULT_RATIO_GROUP,
        candidate_elapsed_group=module.DEFAULT_CANDIDATE_ELAPSED_GROUP,
        reference_target="qpp-cpu",
        candidate_target="mklq-cpu",
    )
    assert weak["status"] == "failed"
    assert "below min_speedup" in "\n".join(weak["failures"])

    dirty = module.check_summary(
        _performance_summary(module, dirty=True),
        required_ratios=list(module.DEFAULT_REQUIRED_RATIOS),
        min_speedup=10.0,
        ratio_group=module.DEFAULT_RATIO_GROUP,
        candidate_elapsed_group=module.DEFAULT_CANDIDATE_ELAPSED_GROUP,
        reference_target="qpp-cpu",
        candidate_target="mklq-cpu",
    )
    assert dirty["status"] == "failed"
    assert "git.dirty is not false" in dirty["failures"]


def test_mklq_performance_evidence_guard_reports_missing_ratio(tmp_path):
    module = _load_performance_evidence_module()
    reports = tmp_path / "reports"
    reports.mkdir()
    summary_path = reports / "local-clean-cpu-test.summary.json"
    summary_path.write_text(json.dumps(
        _performance_summary(module,
                             missing_label="sample_full_register_q20_65536_shots")),
                            encoding="utf-8")

    report = module.build_report(
        root=tmp_path,
        reports=reports,
        pattern="*.summary.json",
        summary_ids=set(),
        required_ratios=list(module.DEFAULT_REQUIRED_RATIOS),
        min_speedup=10.0,
        ratio_group=module.DEFAULT_RATIO_GROUP,
        candidate_elapsed_group=module.DEFAULT_CANDIDATE_ELAPSED_GROUP,
        reference_target="qpp-cpu",
        candidate_target="mklq-cpu",
    )

    assert report["summary"] == {
        "status": "failed",
        "passed": 0,
        "failed": 1,
        "checked": 1,
    }
    assert "missing finite ratio" in "\n".join(
        report["summaries"][0]["failures"])


def _metal_evidence_summary(module,
                            *,
                            evidence_kind=None,
                            performance_scope=None,
                            metal_scope=None,
                            release_flag=True,
                            raw_ignored=True):
    return {
        "schema_version": module.SUMMARY_SCHEMA_VERSION,
        "evidence_kind": evidence_kind or module.DEFAULT_EVIDENCE_KIND,
        "summary_id": "local-metal-test",
        "git": {
            "commit": "abc123",
            "dirty": True,
        },
        "config": {
            "targets": ["qpp-cpu", "mklq-cpu", "mklq-metal"],
            "qubits": [20],
            "repeats": 2,
            "warmups": 1,
            "isolate_rows": True,
        },
        "raw_results": [{
            "path": "benchmarks/mklq/results/local-metal.json",
            "sha256": "b" * 64,
            "status_rows": {
                "ok": 6,
            },
            "tracked": False,
        }],
        "rows": [{
            "target": "mklq-metal",
            "case": "qft-like-state",
            "qubits": 20,
            "shots": 1024,
            "status": "ok",
        }],
        "comparison": {
            "same_day_cross_target_ratio": {
                "qpp_cpu_over_mklq_metal_qft_like_state_q20": 42.0,
            },
            "mklq_metal_elapsed_seconds_median": {
                "qft_like_state_q20": 0.25,
            },
        },
        "interpretation": {
            "clean_worktree": False,
            "raw_json_files_are_ignored": raw_ignored,
            "do_not_treat_as_clean_release_provenance": release_flag,
            "curated_path_labels_are_not_raw_benchmark_fields": True,
            "metal_path_scope": metal_scope or (
                "experimental mklq-metal mixed-path state-vector update "
                "followed by host readback for cudaq.get_state"),
            "performance_claim_scope": performance_scope or (
                "local Apple M5 tuning evidence only; mklq-metal is "
                "experimental mixed-path and this is not release or "
                "cross-machine certification"),
            "summary": "Synthetic local Metal tuning evidence.",
        },
    }


def test_mklq_metal_evidence_guard_accepts_current_summaries():
    module = _load_metal_evidence_module()
    repo_root = Path(__file__).resolve().parents[3]

    report = module.build_report(
        root=repo_root,
        reports=repo_root / "benchmarks" / "mklq" / "reports",
        pattern="*.summary.json",
        summary_ids=set(),
    )

    assert report["summary"]["status"] == "passed"
    assert report["summary"]["checked"] >= 2
    checked_ids = {
        summary["summary_id"]
        for summary in report["summaries"]
    }
    assert "local-metal-y-cy-resident-isolated-q20-2026-06-19" in checked_ids
    assert "local-metal-composite-mixed-path-q20-2026-06-21" in checked_ids
    assert "local-metal-path-labels-q20-2026-06-22" in checked_ids


def test_mklq_metal_evidence_guard_rejects_clean_or_full_native_claims(
        tmp_path):
    module = _load_metal_evidence_module()
    reports = tmp_path / "reports"
    reports.mkdir()
    bad = _metal_evidence_summary(
        module,
        evidence_kind="clean_local_benchmark_evidence",
        performance_scope="release-ready full Metal-native default-ready data",
        metal_scope="full Metal-native backend execution",
        release_flag=False,
        raw_ignored=False,
    )
    (reports / "local-metal-bad.summary.json").write_text(json.dumps(bad),
                                                          encoding="utf-8")

    report = module.build_report(
        root=tmp_path,
        reports=reports,
        pattern="*.summary.json",
        summary_ids=set(),
    )

    assert report["summary"] == {
        "status": "failed",
        "passed": 0,
        "failed": 1,
        "checked": 1,
    }
    failures = "\n".join(report["summaries"][0]["failures"])
    assert "unexpected evidence_kind" in failures
    assert "do_not_treat_as_clean_release_provenance" in failures
    assert "raw JSON files are not marked ignored" in failures
    assert "forbidden Metal claim" in failures


def test_mklq_metal_evidence_guard_rejects_missing_declared_static_labels(
        tmp_path):
    module = _load_metal_evidence_module()
    reports = tmp_path / "reports"
    reports.mkdir()
    bad = _metal_evidence_summary(module)
    bad["interpretation"]["metal_path_labels_are_static_case_map"] = True
    (reports / "local-metal-missing-labels.summary.json").write_text(
        json.dumps(bad), encoding="utf-8")

    report = module.build_report(
        root=tmp_path,
        reports=reports,
        pattern="*.summary.json",
        summary_ids=set(),
    )

    assert report["summary"] == {
        "status": "failed",
        "passed": 0,
        "failed": 1,
        "checked": 1,
    }
    failures = "\n".join(report["summaries"][0]["failures"])
    assert "lacks metal_path_label" in failures
    assert "unexpected metal_path_label_source" in failures
    assert "lacks metal_path_scope" in failures
    assert "lacks metal_evidence_boundary" in failures


def test_mklq_metal_runtime_counter_probe_selects_counter_tests():
    module = _load_metal_runtime_counter_probe_module()
    listing = """
Test project /repo/build-python
  Test #787: mklq_metal_MKLQMetalTester.MetalRuntimeKeepsResidentStateAcrossGateSequence
  Test #789: mklq_metal_MKLQMetalTester.MetalRuntimeFillsResidentProbabilitiesWithoutStateReadback
  Test #792: mklq_metal_MKLQMetalTester.SimulatorKeepsSupportedGateSequenceResidentUntilReadback
  Test #795: mklq_metal_MKLQMetalTester.SimulatorSamplesResidentDenseStateWithoutReadback
  Test #800: mklq_metal_MKLQMetalTester.SimulatorMeasuresAndResetsResidentStateWithoutReadback
  Test #999: qpp_MKLQMetalTester.Unrelated
"""

    tests = module.select_counter_tests(listing)

    assert tests == [
        "mklq_metal_MKLQMetalTester.MetalRuntimeKeepsResidentStateAcrossGateSequence",
        "mklq_metal_MKLQMetalTester.MetalRuntimeFillsResidentProbabilitiesWithoutStateReadback",
        "mklq_metal_MKLQMetalTester.SimulatorKeepsSupportedGateSequenceResidentUntilReadback",
        "mklq_metal_MKLQMetalTester.SimulatorSamplesResidentDenseStateWithoutReadback",
        "mklq_metal_MKLQMetalTester.SimulatorMeasuresAndResetsResidentStateWithoutReadback",
    ]


def test_mklq_metal_runtime_counter_probe_builds_bounded_report(monkeypatch,
                                                                tmp_path):
    module = _load_metal_runtime_counter_probe_module()
    commands = []
    expected_test_names = [
        module.TEST_PREFIX + suffix for suffix in module.COUNTER_TEST_SUFFIXES
    ]

    def fake_command_output(cwd, command):
        commands.append(command)
        if command[:3] == ["ctest", "--test-dir", str(tmp_path)]:
            lines = ["Test project /tmp/build"]
            lines.extend(
                f"  Test #{787 + index}: {name}"
                for index, name in enumerate(expected_test_names))
            return "\n".join(lines)
        return "ignored"

    def fake_run_command(cwd, command):
        commands.append(command)
        return {
            "returncode": 0,
            "stdout": "counter assertions passed",
            "stderr": "",
        }

    monkeypatch.setattr(module, "command_output", fake_command_output)
    monkeypatch.setattr(module, "run_command", fake_run_command)

    report = module.build_report(repo_root=tmp_path, build_dir=tmp_path)

    assert report["schema_version"] == (
        "mklq-metal-runtime-counter-probe-v1")
    assert report["evidence_kind"] == "local_runtime_counter_probe"
    assert report["summary"] == {
        "status": "passed",
        "selected": len(expected_test_names),
        "expected": len(expected_test_names),
        "missing": 0,
        "passed": len(expected_test_names),
        "failed": 0,
    }
    assert report["expected_counter_tests"] == expected_test_names
    assert report["missing_counter_tests"] == []
    assert report["boundary"]["runtime_counter_evidence"] is True
    assert module.COUNTER_SOURCE in report["boundary"][
        "runtime_counter_source"]
    assert report["boundary"]["release_signoff"] is False
    assert report["boundary"]["all_metal_execution_proof"] is False
    assert [test["status"] for test in report["tests"]] == (
        ["passed"] * len(expected_test_names))
    assert all(test["counter_source"] == module.COUNTER_SOURCE
               for test in report["tests"])
    assert all("stdout" not in test for test in report["tests"])
    run_commands = [command for command in commands if command[:1] == ["ctest"]
                    and "--output-on-failure" in command]
    assert len(run_commands) == len(expected_test_names)
    assert run_commands[0][:5] == [
        "ctest",
        "--test-dir",
        str(tmp_path),
        "-R",
        r"^mklq_metal_MKLQMetalTester\.MetalRuntimeKeepsResidentStateAcrossGateSequence$",
    ]


def test_mklq_metal_runtime_counter_probe_fails_when_expected_tests_missing(
        monkeypatch, tmp_path):
    module = _load_metal_runtime_counter_probe_module()

    def fake_command_output(cwd, command):
        return """
Test project /tmp/build
  Test #787: mklq_metal_MKLQMetalTester.MetalRuntimeKeepsResidentStateAcrossGateSequence
"""

    def fake_run_command(cwd, command):
        return {
            "returncode": 0,
            "stdout": "counter assertions passed",
            "stderr": "",
        }

    monkeypatch.setattr(module, "command_output", fake_command_output)
    monkeypatch.setattr(module, "run_command", fake_run_command)

    report = module.build_report(repo_root=tmp_path, build_dir=tmp_path)

    assert report["summary"]["status"] == "failed"
    assert report["summary"]["selected"] == 1
    assert report["summary"]["expected"] == len(module.COUNTER_TEST_SUFFIXES)
    assert report["summary"]["missing"] == len(module.COUNTER_TEST_SUFFIXES) - 1
    assert (module.TEST_PREFIX +
            "SimulatorSamplesDenseFullRegisterThroughMetalProbabilityFill"
            in report["missing_counter_tests"])


def test_mklq_metal_runtime_counter_probe_records_per_test_failures(
        monkeypatch, tmp_path):
    module = _load_metal_runtime_counter_probe_module()
    expected_test_names = [
        module.TEST_PREFIX + suffix for suffix in module.COUNTER_TEST_SUFFIXES
    ]
    failing_test = expected_test_names[1]
    run_commands = []

    def fake_command_output(cwd, command):
        lines = ["Test project /tmp/build"]
        lines.extend(
            f"  Test #{787 + index}: {name}"
            for index, name in enumerate(expected_test_names))
        return "\n".join(lines)

    def fake_run_command(cwd, command):
        run_commands.append(command)
        regex = command[command.index("-R") + 1]
        return {
            "returncode": 1 if failing_test.replace(".", r"\.") in regex else 0,
            "stdout": f"{regex} stdout",
            "stderr": f"{regex} stderr",
        }

    monkeypatch.setattr(module, "command_output", fake_command_output)
    monkeypatch.setattr(module, "run_command", fake_run_command)

    report = module.build_report(repo_root=tmp_path, build_dir=tmp_path)

    assert report["summary"]["status"] == "failed"
    assert report["summary"]["passed"] == len(expected_test_names) - 1
    assert report["summary"]["failed"] == 1
    failed_tests = [
        test for test in report["tests"] if test["status"] == "failed"
    ]
    assert [test["name"] for test in failed_tests] == [failing_test]
    assert "failure_excerpt" in failed_tests[0]
    assert len(run_commands) == len(expected_test_names)


def _runtime_counter_summary_fixture():
    tests = [
        "mklq_metal_MKLQMetalTester.SimulatorKeepsSupportedGateSequenceResidentUntilReadback",
        "mklq_metal_MKLQMetalTester.SimulatorSamplesResidentDenseStateWithoutReadback",
        "mklq_metal_MKLQMetalTester.SimulatorMeasuresAndResetsResidentStateWithoutReadback",
        "mklq_metal_MKLQMetalTester.SimulatorReuploadsResidentStateAfterUnsupportedGateFallback",
    ]
    return {
        "schema_version": "mklq-metal-runtime-counter-probe-v1",
        "evidence_kind": "local_runtime_counter_probe",
        "created_at_utc": "2026-06-22T00:00:00+00:00",
        "summary": {
            "status": "passed",
            "expected": len(tests),
            "selected": len(tests),
            "missing": 0,
            "passed": len(tests),
            "failed": 0,
        },
        "boundary": {
            "runtime_counter_evidence": True,
            "release_signoff": False,
            "all_metal_execution_proof": False,
            "raw_logs_truncated": True,
        },
        "expected_counter_tests": tests,
        "missing_counter_tests": [],
        "tests": [{
            "name": name,
            "status": "passed",
        } for name in tests],
    }


def test_mklq_metal_runtime_counter_summary_groups_counter_coverage(tmp_path):
    module = _load_metal_runtime_counter_summary_module()
    report = tmp_path / "probe.counter.json"
    report.write_text(json.dumps(_runtime_counter_summary_fixture()),
                      encoding="utf-8")

    summary = module.build_summary([report])

    assert summary["schema_version"] == (
        "mklq-metal-runtime-counter-summary-v1")
    assert summary["summary"] == {
        "status": "passed",
        "report_count": 1,
        "expected": 4,
        "selected": 4,
        "missing": 0,
        "passed": 4,
        "failed": 0,
    }
    assert summary["boundary"]["runtime_counter_evidence"] is True
    assert summary["boundary"]["release_signoff"] is False
    assert summary["boundary"]["all_metal_execution_proof"] is False
    categories = {
        category["category"]: category
        for category in summary["categories"]
    }
    assert categories["resident_gate"]["passed"] == 1
    assert categories["probability_sampling"]["passed"] == 1
    assert categories["measurement_reset"]["passed"] == 1
    assert categories["fallback_boundary"]["passed"] == 1


def test_mklq_metal_runtime_counter_summary_renders_markdown(tmp_path):
    module = _load_metal_runtime_counter_summary_module()
    report = tmp_path / "probe.counter.json"
    report.write_text(json.dumps(_runtime_counter_summary_fixture()),
                      encoding="utf-8")

    markdown = module.render_markdown(module.build_summary([report]))

    assert "# MKL-Q Metal Runtime Counter Summary" in markdown
    assert "fallback_boundary" in markdown
    assert "runtime counter evidence" in markdown
    assert "not release sign-off" in markdown
    assert "not proof that every operation stayed on Metal" in markdown


def test_mklq_metal_runtime_counter_summary_fails_unsafe_boundaries(tmp_path):
    module = _load_metal_runtime_counter_summary_module()
    payload = _runtime_counter_summary_fixture()
    payload["boundary"]["runtime_counter_evidence"] = False
    payload["boundary"]["release_signoff"] = True
    payload["boundary"]["all_metal_execution_proof"] = True
    report = tmp_path / "probe.counter.json"
    report.write_text(json.dumps(payload), encoding="utf-8")

    summary = module.build_summary([report])

    assert summary["summary"]["status"] == "failed"
    assert summary["boundary"]["runtime_counter_evidence"] is False
    assert summary["boundary"]["release_signoff"] is True
    assert summary["boundary"]["all_metal_execution_proof"] is True


def test_mklq_metal_runtime_counter_summary_cli_writes_markdown(monkeypatch,
                                                                 tmp_path):
    module = _load_metal_runtime_counter_summary_module()
    report = tmp_path / "probe.counter.json"
    output = tmp_path / "summary.md"
    report.write_text(json.dumps(_runtime_counter_summary_fixture()),
                      encoding="utf-8")

    monkeypatch.setattr(module.sys, "argv", [
        "summarize_metal_runtime_counters.py",
        "--reports",
        str(tmp_path),
        "--output",
        str(output),
    ])

    exit_code = module.main()

    assert exit_code == 0
    assert output.exists()
    assert "MKL-Q Metal Runtime Counter Summary" in output.read_text(
        encoding="utf-8")


def test_mklq_public_healthcheck_parses_metal_runtime_counter_probe(tmp_path):
    module = _load_public_healthcheck_module()
    config = _public_healthcheck_config(module, tmp_path)
    report_dir = tmp_path / "benchmarks" / "mklq" / "reports"
    report_dir.mkdir(parents=True)
    counter_report = report_dir / "probe.counter.json"
    expected_tests = [
        "mklq_metal_MKLQMetalTester.CounterA",
        "mklq_metal_MKLQMetalTester.CounterB",
    ]
    counter_report.write_text(json.dumps({
        "schema_version": "mklq-metal-runtime-counter-probe-v1",
        "evidence_kind": "local_runtime_counter_probe",
        "summary": {
            "status": "passed",
            "expected": 2,
            "selected": 2,
            "missing": 0,
            "passed": 2,
            "failed": 0,
        },
        "boundary": {
            "runtime_counter_evidence": True,
            "release_signoff": False,
            "all_metal_execution_proof": False,
        },
        "expected_counter_tests": expected_tests,
        "missing_counter_tests": [],
        "tests": [{
            "name": expected_tests[0],
            "status": "passed",
        }, {
            "name": expected_tests[1],
            "status": "passed",
        }],
    }),
                              encoding="utf-8")

    result = module.run_metal_runtime_counter_probe_parse(config)

    assert result["status"] == "passed"
    assert result["details"]["counter_report_count"] == 1
    assert result["details"]["expected_tests"] == 2
    assert result["details"]["selected_tests"] == 2
    assert result["details"]["missing_tests"] == 0

    counter_report.write_text(json.dumps({
        "schema_version": "mklq-metal-runtime-counter-probe-v1",
        "evidence_kind": "local_runtime_counter_probe",
        "summary": {
            "status": "passed",
            "expected": 2,
            "selected": 1,
            "missing": 1,
            "passed": 1,
            "failed": 0,
        },
        "boundary": {
            "runtime_counter_evidence": True,
            "release_signoff": False,
            "all_metal_execution_proof": True,
        },
        "expected_counter_tests": expected_tests,
        "missing_counter_tests": [expected_tests[1]],
        "tests": [{
            "name": expected_tests[0],
            "status": "passed",
            "stdout": "raw log leak",
        }],
    }),
                              encoding="utf-8")

    result = module.run_metal_runtime_counter_probe_parse(config)

    assert result["status"] == "failed"
    assert "all_metal_execution_proof" in "\n".join(
        result["details"]["failures"])
    assert "missing counter test count" in "\n".join(
        result["details"]["failures"])
    assert "raw stdout" in "\n".join(result["details"]["failures"])


def test_mklq_public_healthcheck_runs_metal_evidence_guard(monkeypatch,
                                                           tmp_path):
    module = _load_public_healthcheck_module()
    config = _public_healthcheck_config(module, tmp_path)
    calls = []

    def fake_run_command(config, command, env_overlay=None):
        calls.append(command)
        return {
            "returncode": 0,
            "command": command,
            "stdout_tail": "{}",
            "stderr_tail": "",
        }

    monkeypatch.setattr(module, "run_command", fake_run_command)

    result = module.run_metal_evidence_check(config)

    assert result["status"] == "passed"
    assert calls[0][1].endswith("benchmarks/mklq/check_metal_evidence.py")
    assert "--reports" in calls[0]


def test_mklq_public_healthcheck_plan_includes_metal_evidence_guard(tmp_path):
    module = _load_public_healthcheck_module()
    config = _public_healthcheck_config(module,
                                        tmp_path,
                                        include_harness_tests=False,
                                        plan_only=True)

    steps = [step.name for step in module.build_steps(config)]

    assert steps.index("performance_evidence_guard") < steps.index(
        "metal_evidence_guard")
    assert steps.index("metal_evidence_guard") < steps.index(
        "metal_runtime_counter_probe_parse")
    assert steps.index("metal_runtime_counter_probe_parse") < steps.index(
        "benchmark_helper_py_compile")


def test_mklq_public_healthcheck_writes_json_report(monkeypatch, tmp_path):
    module = _load_public_healthcheck_module()
    config = _public_healthcheck_config(module, tmp_path)

    monkeypatch.setattr(
        module, "build_steps",
        lambda config: [
            module.Step("ok", "passing synthetic step",
                        lambda config: module.passed({"value": 1})),
            module.Step("bad", "failing synthetic step",
                        lambda config: module.failed("synthetic failure")),
        ])

    report = module.run_healthcheck(config)
    written = json.loads(config.output.read_text(encoding="utf-8"))

    assert report["summary"] == {"status": "failed", "passed": 1, "failed": 1}
    assert written["summary"] == report["summary"]
    assert [step["name"] for step in written["steps"]] == ["ok", "bad"]
    assert written["steps"][0]["details"] == {"value": 1}
    assert written["steps"][1]["message"] == "synthetic failure"


def _readiness_repo_payload():
    return {
        "nameWithOwner": "wuls968/MKL-Q",
        "isFork": True,
        "parent": {
            "name": "cuda-quantum",
            "owner": {
                "login": "NVIDIA",
            },
        },
        "defaultBranchRef": {
            "name": "main",
        },
        "description": (
            "CUDA-Q-compatible Apple Silicon simulator fork with MKL-Q targets"
        ),
        "repositoryTopics": [{
            "name": "accelerate",
        }, {
            "name": "apple-silicon",
        }, {
            "name": "cuda-quantum",
        }, {
            "name": "metal",
        }, {
            "name": "mklq",
        }, {
            "name": "quantum-computing",
        }],
        "licenseInfo": {
            "key": "apache-2.0",
        },
        "visibility": "PUBLIC",
        "url": "https://github.com/wuls968/MKL-Q",
    }


def test_mklq_public_readiness_audit_builds_passing_report(monkeypatch,
                                                           tmp_path):
    module = _load_public_readiness_audit_module()
    config = module.AuditConfig(
        repo_root=tmp_path,
        repo="wuls968/MKL-Q",
        workflow="MKL-Q public hygiene",
        output=tmp_path / "readiness.json",
    )
    calls = []

    def fake_command_output(cwd, command):
        calls.append(command)
        if command == ["git", "status", "--short", "--branch"]:
            return "## main...origin/main"
        if command == ["git", "rev-parse", "--is-shallow-repository"]:
            return "false"
        if command == ["git", "rev-parse", "HEAD"]:
            return "abc123"
        if command == ["git", "ls-remote", "origin", "refs/heads/main"]:
            return "abc123\trefs/heads/main"
        if command == ["git", "ls-remote", "--tags", "origin", "refs/tags/*"]:
            return ""
        if command == ["git", "ls-files"]:
            return "\n".join([
                "README.md",
                ".github/workflows/mklq-public-hygiene.yml",
                "docs/mklq/public-readiness.md",
            ])
        if command == ["git", "ls-files", ".github/workflows"]:
            return ".github/workflows/mklq-public-hygiene.yml"
        if command[:3] == ["gh", "repo", "view"]:
            return json.dumps(_readiness_repo_payload())
        if command[:3] == ["gh", "api", "repos/wuls968/MKL-Q/branches/main"]:
            return json.dumps({
                "name": "main",
                "protected": True,
                "commit": "abc123",
            })
        if command[:3] == [
                "gh", "api",
                "repos/wuls968/MKL-Q/branches/main/protection"
        ]:
            return json.dumps({
                "required_status_checks": {
                    "strict": True,
                    "contexts": ["Source-only repository checks"],
                },
                "allow_force_pushes": {
                    "enabled": False,
                },
                "allow_deletions": {
                    "enabled": False,
                },
                "enforce_admins": {
                    "enabled": True,
                },
            })
        if command[:3] == ["gh", "run", "list"]:
            return json.dumps([{
                "status": "completed",
                "conclusion": "success",
                "headSha": "abc123",
                "url": "https://github.com/wuls968/MKL-Q/actions/runs/1",
            }])
        if command[:3] == ["gh", "release", "list"]:
            return ""
        raise AssertionError(command)

    monkeypatch.setattr(module, "command_output", fake_command_output)

    report = module.build_report(config)

    assert report["schema_version"] == "mklq-public-readiness-audit-v1"
    assert report["summary"]["status"] == "passed"
    assert report["summary"]["failed"] == 0
    checks = {check["name"]: check for check in report["checks"]}
    assert checks["github_repository"]["status"] == "passed"
    assert checks["branch_protection"]["status"] == "passed"
    assert checks["latest_public_hygiene"]["details"]["headSha"] == "abc123"
    assert any(call[:3] == ["gh", "repo", "view"] for call in calls)


def test_mklq_public_readiness_audit_rejects_release_tags_and_unprotected_main(
        monkeypatch, tmp_path):
    module = _load_public_readiness_audit_module()
    config = module.AuditConfig(
        repo_root=tmp_path,
        repo="wuls968/MKL-Q",
        workflow="MKL-Q public hygiene",
        output=tmp_path / "readiness.json",
    )

    def fake_command_output(cwd, command):
        if command == ["git", "status", "--short", "--branch"]:
            return "## main...origin/main"
        if command == ["git", "rev-parse", "--is-shallow-repository"]:
            return "false"
        if command == ["git", "rev-parse", "HEAD"]:
            return "abc123"
        if command == ["git", "ls-remote", "origin", "refs/heads/main"]:
            return "abc123\trefs/heads/main"
        if command == ["git", "ls-remote", "--tags", "origin", "refs/tags/*"]:
            return "abc123\trefs/tags/v0.1.0"
        if command == ["git", "ls-files"]:
            return "README.md"
        if command == ["git", "ls-files", ".github/workflows"]:
            return ".github/workflows/mklq-public-hygiene.yml"
        if command[:3] == ["gh", "repo", "view"]:
            return json.dumps(_readiness_repo_payload())
        if command[:3] == ["gh", "api", "repos/wuls968/MKL-Q/branches/main"]:
            return json.dumps({
                "name": "main",
                "protected": False,
                "commit": "abc123",
            })
        if command[:3] == [
                "gh", "api",
                "repos/wuls968/MKL-Q/branches/main/protection"
        ]:
            return json.dumps({
                "required_status_checks": {
                    "strict": False,
                    "contexts": [],
                },
                "allow_force_pushes": {
                    "enabled": True,
                },
                "allow_deletions": {
                    "enabled": True,
                },
                "enforce_admins": {
                    "enabled": False,
                },
            })
        if command[:3] == ["gh", "run", "list"]:
            return json.dumps([{
                "status": "completed",
                "conclusion": "success",
                "headSha": "abc123",
            }])
        if command[:3] == ["gh", "release", "list"]:
            return ""
        raise AssertionError(command)

    monkeypatch.setattr(module, "command_output", fake_command_output)

    report = module.build_report(config)
    failures = "\n".join(check.get("message", "") for check in report["checks"]
                         if check["status"] == "failed")

    assert report["summary"]["status"] == "failed"
    assert "branch is not protected" in failures
    assert "administrator enforcement is not enabled" in failures
    assert "release tags exist" in failures


def _preflight_config(module, tmp_path, *, require_clean=True, check_github=True):
    return module.PreflightConfig(
        repo_root=tmp_path,
        repo="wuls968/MKL-Q",
        output=tmp_path / "preflight.json",
        require_clean=require_clean,
        check_github=check_github,
    )


def _preflight_branch_protection(*, protected=True, strict=True,
                                 enforce_admins=True,
                                 allow_force_pushes=False,
                                 allow_deletions=False,
                                 contexts=None):
    contexts = ["Source-only repository checks"] if contexts is None else contexts
    return {
        "branch": {
            "name": "main",
            "protected": protected,
        },
        "protection": {
            "required_status_checks": {
                "strict": strict,
                "contexts": contexts,
                "checks": [],
            },
            "allow_force_pushes": {
                "enabled": allow_force_pushes,
            },
            "allow_deletions": {
                "enabled": allow_deletions,
            },
            "enforce_admins": {
                "enabled": enforce_admins,
            },
        },
    }


def test_mklq_preflight_audit_builds_passing_report(monkeypatch, tmp_path):
    module = _load_preflight_audit_module()
    config = _preflight_config(module, tmp_path)
    (tmp_path / ".git").mkdir()
    protection = _preflight_branch_protection()
    calls = []

    def fake_command_output(cwd, command):
        calls.append(command)
        if command == ["git", "status", "--short", "--branch"]:
            return "## codex/topic...origin/codex/topic"
        if command == ["git", "rev-parse", "--is-shallow-repository"]:
            return "false"
        if command == ["git", "rev-parse", "--git-dir"]:
            return ".git"
        if command == ["git", "remote", "-v"]:
            return "\n".join([
                "origin\thttps://github.com/wuls968/MKL-Q.git (fetch)",
                "origin\thttps://github.com/wuls968/MKL-Q.git (push)",
                "upstream\thttps://github.com/NVIDIA/cuda-quantum.git (fetch) [blob:none]",
                "upstream\thttps://github.com/NVIDIA/cuda-quantum.git (push)",
            ])
        if command == ["git", "ls-files"]:
            return "\n".join([
                "README.md",
                ".github/workflows/mklq-public-hygiene.yml",
            ])
        if command == ["git", "status", "--ignored", "--short"]:
            return "\n".join([
                "!! benchmarks/mklq/results/",
                "!! build-python/",
                "!! .DS_Store",
            ])
        if command == [
                "gh", "api", "repos/wuls968/MKL-Q/branches/main"
        ]:
            return json.dumps(protection["branch"])
        if command == [
                "gh", "api", "repos/wuls968/MKL-Q/branches/main/protection"
        ]:
            return json.dumps(protection["protection"])
        raise AssertionError(command)

    monkeypatch.setattr(module, "command_output", fake_command_output)

    report = module.build_report(config)

    assert report["schema_version"] == "mklq-preflight-audit-v1"
    assert report["summary"] == {
        "status": "passed",
        "passed": 5,
        "failed": 0,
    }
    checks = {check["name"]: check for check in report["checks"]}
    assert checks["git_locks"]["details"]["lock_files"] == []
    assert checks["ignored_local_artifacts"]["details"]["ignored_count"] == 3
    assert checks["branch_protection"]["status"] == "passed"
    assert any(call[:2] == ["gh", "api"] for call in calls)


def test_mklq_preflight_audit_retries_transient_git_locks(monkeypatch,
                                                          tmp_path):
    module = _load_preflight_audit_module()
    config = _preflight_config(module, tmp_path, check_github=False)
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    lock = git_dir / "index.lock"
    lock.write_text("transient lock", encoding="utf-8")
    checks = 0

    def fake_command_output(cwd, command):
        if command == ["git", "status", "--short", "--branch"]:
            return "## codex/topic...origin/codex/topic"
        if command == ["git", "rev-parse", "--is-shallow-repository"]:
            return "false"
        if command == ["git", "rev-parse", "--git-dir"]:
            return ".git"
        if command == ["git", "remote", "-v"]:
            return "\n".join([
                "origin\thttps://github.com/wuls968/MKL-Q.git (fetch)",
                "upstream\thttps://github.com/NVIDIA/cuda-quantum.git (fetch)",
            ])
        if command == ["git", "ls-files"]:
            return "README.md"
        if command == ["git", "status", "--ignored", "--short"]:
            return ""
        raise AssertionError(command)

    def fake_sleep(_seconds):
        nonlocal checks
        checks += 1
        lock.unlink()

    monkeypatch.setattr(module, "command_output", fake_command_output)
    monkeypatch.setattr(module.time, "sleep", fake_sleep)

    report = module.build_report(config)
    lock_check = {
        check["name"]: check for check in report["checks"]
    }["git_locks"]

    assert report["summary"]["status"] == "passed"
    assert lock_check["details"]["lock_files"] == []
    assert lock_check["details"]["rechecks"] == 1
    assert checks == 1


def test_mklq_preflight_audit_rejects_locks_raw_artifacts_and_bad_protection(
        monkeypatch, tmp_path):
    module = _load_preflight_audit_module()
    config = _preflight_config(module, tmp_path)
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "index.lock").write_text("stale lock", encoding="utf-8")
    protection = _preflight_branch_protection(
        protected=False,
        strict=False,
        enforce_admins=False,
        allow_force_pushes=True,
        allow_deletions=True,
        contexts=[],
    )

    def fake_command_output(cwd, command):
        if command == ["git", "status", "--short", "--branch"]:
            return "## main...origin/main\n M docs/mklq/validation.md"
        if command == ["git", "rev-parse", "--is-shallow-repository"]:
            return "false"
        if command == ["git", "rev-parse", "--git-dir"]:
            return ".git"
        if command == ["git", "remote", "-v"]:
            return "origin\thttps://github.com/wuls968/MKL-Q.git (fetch)"
        if command == ["git", "ls-files"]:
            return "\n".join([
                "README.md",
                "benchmarks/mklq/results/raw.json",
                ".DS_Store",
            ])
        if command == ["git", "status", "--ignored", "--short"]:
            return ""
        if command == [
                "gh", "api", "repos/wuls968/MKL-Q/branches/main"
        ]:
            return json.dumps(protection["branch"])
        if command == [
                "gh", "api", "repos/wuls968/MKL-Q/branches/main/protection"
        ]:
            return json.dumps(protection["protection"])
        raise AssertionError(command)

    monkeypatch.setattr(module, "command_output", fake_command_output)

    report = module.build_report(config)
    failures = "\n".join(check.get("message", "") for check in report["checks"]
                         if check["status"] == "failed")

    assert report["summary"]["status"] == "failed"
    assert "working tree is dirty" in failures
    assert "git lock files are present" in failures
    assert "generated or local artifacts are tracked" in failures
    assert "upstream remote is missing" in failures
    assert "branch is not protected" in failures
    assert "required status check is missing" in failures
    assert "administrator enforcement is not enabled" in failures


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
            "status": "ok",
            "target": "mklq-metal",
            "case": "y-state",
            "shots": 1024,
            "metal_path_label":
                "mklq_metal_resident_single_gate_state_host_readback",
            "metal_path_scope":
                "resident fp32 Metal single-target gate update followed by "
                "host readback for cudaq.get_state",
            "metal_path_label_source": "benchmark_harness_static_case_map",
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
    assert "## Metal Path Labels" in markdown
    assert "mklq_metal_resident_single_gate_state_host_readback" in markdown
    assert "benchmark_harness_static_case_map" in markdown


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


def test_mklq_benchmark_dry_run_records_metal_path_metadata(tmp_path):
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "benchmarks" / "mklq" / "bench_mklq_targets.py"
    output = tmp_path / "dry-run-metal-paths.json"
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)

    subprocess.run([
        sys.executable,
        str(script),
        "--dry-run",
        "--targets",
        "mklq-metal",
        "--cases",
        "y-state,cy-state,qft-like-state,sample-full-register",
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
    rows = {
        row["case"]: row
        for row in report["results"]
    }

    assert rows["y-state"]["metrics"]["metal_path_label"] == (
        "mklq_metal_resident_single_gate_state_host_readback")
    assert rows["cy-state"]["metrics"]["metal_path_label"] == (
        "mklq_metal_resident_controlled_gate_state_host_readback")
    assert rows["qft-like-state"]["metrics"]["metal_path_label"] == (
        "mklq_metal_mixed_composite_state_host_readback")
    assert rows["sample-full-register"]["metrics"]["metal_path_label"] == (
        "mklq_metal_mixed_sampling_host_counts")
    for row in rows.values():
        metrics = row["metrics"]
        assert metrics["metal_path_label_source"] == (
            "benchmark_harness_static_case_map")
        assert metrics["metal_full_native"] is False
        assert metrics["metal_runtime_counter"] is False
        assert "not a runtime counter" in metrics["metal_evidence_boundary"]


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


def test_mklq_benchmark_run_case_records_metal_path_metrics(monkeypatch):
    module = _load_benchmark_module()

    class FakeKernel:

        def qalloc(self, qubits):
            return list(range(qubits))

        def ry(self, theta, target):
            pass

        def rz(self, theta, target):
            pass

        def y(self, target):
            pass

    class FakeCudaq:

        def reset_target(self):
            pass

        def set_target(self, target):
            assert target == "mklq-metal"

        def set_random_seed(self, seed):
            assert seed == 13

        def make_kernel(self):
            return FakeKernel()

        def get_state(self, kernel):
            return object()

    def fake_timed_repeats(action, repeats):
        action()
        return [0.25]

    monkeypatch.setattr(module, "timed_repeats", fake_timed_repeats)
    monkeypatch.setattr(module, "process_max_rss_bytes", lambda: 4096)

    row = module.run_case(FakeCudaq(),
                          "mklq-metal",
                          "y-state",
                          qubits=3,
                          shots=16,
                          repeats=1,
                          warmups=0,
                          layers=2)

    assert row["status"] == "ok"
    metrics = row["metrics"]
    assert metrics["metal_path_label"] == (
        "mklq_metal_resident_single_gate_state_host_readback")
    assert metrics["metal_path_scope"] == (
        "resident fp32 Metal single-target gate update followed by host "
        "readback for cudaq.get_state")
    assert metrics["metal_path_label_source"] == (
        "benchmark_harness_static_case_map")
    assert metrics["metal_full_native"] is False
    assert metrics["metal_runtime_counter"] is False


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


def test_mklq_benchmark_dry_run_accepts_composite_state_cases(tmp_path):
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "benchmarks" / "mklq" / "bench_mklq_targets.py"
    output = tmp_path / "dry-run-composite-state.json"
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)

    subprocess.run([
        sys.executable,
        str(script),
        "--dry-run",
        "--targets",
        "mklq-cpu",
        "--cases",
        "qft-like-state,seeded-clifford-state",
        "--qubits",
        "4",
        "--layers",
        "2",
        "--output",
        str(output),
    ],
                   check=True,
                   capture_output=True,
                   text=True,
                   env=env)

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["config"]["cases"] == [
        "qft-like-state",
        "seeded-clifford-state",
    ]
    rows = report["results"]
    assert len(rows) == 2
    assert {row["status"] for row in rows} == {"planned"}
    assert {row["case"] for row in rows} == {
        "qft-like-state",
        "seeded-clifford-state",
    }
    assert {row["estimated_state_bytes"] for row in rows} == {16 * (1 << 4)}


def test_mklq_benchmark_qft_like_case_records_composite_metrics(monkeypatch):
    module = _load_benchmark_module()

    class FakeKernel:

        def __init__(self):
            self.operations = []

        def qalloc(self, qubits):
            return list(range(qubits))

        def x(self, target):
            self.operations.append(("x", target))

        def h(self, target):
            self.operations.append(("h", target))

        def crz(self, theta, control, target):
            self.operations.append(("crz", theta, control, target))

        def swap(self, left, right):
            self.operations.append(("swap", left, right))

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
    monkeypatch.setattr(module, "process_max_rss_bytes", lambda: 16384)

    fake_cudaq = FakeCudaq()
    row = module.run_case(fake_cudaq,
                          "mklq-cpu",
                          "qft-like-state",
                          qubits=4,
                          shots=16,
                          repeats=1,
                          warmups=0,
                          layers=2)

    assert row["status"] == "ok"
    metrics = row["metrics"]
    assert metrics["state_prep_gate_count"] == 2
    assert metrics["qft_h_gate_count"] == 8
    assert metrics["qft_crz_gate_count"] == 12
    assert metrics["qft_swap_gate_count"] == 4
    assert metrics["qft_like_gate_count"] == 24
    assert metrics["gate_count"] == 26
    assert metrics["layers"] == 2
    assert metrics["qft_like_state_throughput_per_second"] == 48
    assert metrics["process_max_rss_bytes_cumulative"] == 16384
    operations = fake_cudaq.kernels[0].operations
    assert operations.count(("x", 0)) == 1
    assert operations.count(("x", 3)) == 1
    assert sum(1 for op in operations if op[0] == "crz") == 12


def test_mklq_benchmark_seeded_clifford_case_records_composite_metrics(
        monkeypatch):
    module = _load_benchmark_module()

    class FakeKernel:

        def __init__(self):
            self.operations = []

        def qalloc(self, qubits):
            return list(range(qubits))

        def h(self, target):
            self.operations.append(("h", target))

        def s(self, target):
            self.operations.append(("s", target))

        def sdg(self, target):
            self.operations.append(("sdg", target))

        def x(self, target):
            self.operations.append(("x", target))

        def y(self, target):
            self.operations.append(("y", target))

        def z(self, target):
            self.operations.append(("z", target))

        def cx(self, control, target):
            self.operations.append(("cx", control, target))

        def cy(self, control, target):
            self.operations.append(("cy", control, target))

        def cz(self, control, target):
            self.operations.append(("cz", control, target))

        def swap(self, left, right):
            self.operations.append(("swap", left, right))

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
    monkeypatch.setattr(module, "process_max_rss_bytes", lambda: 32768)

    fake_cudaq = FakeCudaq()
    row = module.run_case(fake_cudaq,
                          "mklq-cpu",
                          "seeded-clifford-state",
                          qubits=4,
                          shots=16,
                          repeats=1,
                          warmups=0,
                          layers=2)

    assert row["status"] == "ok"
    metrics = row["metrics"]
    assert metrics["seeded_clifford_single_gate_count"] == 8
    assert metrics["seeded_clifford_two_qubit_gate_count"] == 8
    assert metrics["seeded_clifford_gate_count"] == 16
    assert metrics["seeded_clifford_seed"] == 17
    assert metrics["gate_count"] == 16
    assert metrics["layers"] == 2
    assert metrics["seeded_clifford_state_throughput_per_second"] == 64
    assert metrics["process_max_rss_bytes_cumulative"] == 32768
    operations = fake_cudaq.kernels[0].operations
    assert len(operations) == 16
    assert sum(1 for op in operations if op[0] in {"cx", "cy", "cz",
                                                   "swap"}) == 8


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
        "34f4b260d1c657ad626c526eed4e6b9d3a441be4")
    assert summary["git"]["dirty"] is False
    assert summary["interpretation"]["clean_worktree"] is True
    assert summary["raw_results"][0]["sha256"] == (
        "2b438094b63bf0dda2a06be2785b75ca54fdb2c8b2fa74d6ba212b6fea832ef0")
    assert summary["raw_results"][0]["status_rows"] == {"ok": 6}
    assert summary["raw_results"][1]["sha256"] == (
        "b07b3ba92b83c0db12ad560ab650e3be035f543fb690dcb5ff946852e6eb423f")
    assert summary["raw_results"][1]["status_rows"] == {"ok": 4}
    assert summary["raw_results"][2]["sha256"] == (
        "167b5c4adef8fa0da682e05c841f0475da0570bc50483739b71b8d6fcab2716a")
    assert summary["raw_results"][2]["status_rows"] == {"ok": 8}
    assert summary["machine"]["cpu_brand"] == "Apple M5"
    assert summary["config"]["targets"] == ["qpp-cpu", "mklq-cpu"]
    assert summary["config"]["cases"] == [
        "y-state", "cy-state", "cz-state", "qft-like-state",
        "seeded-clifford-state", "sample-full-register",
        "sample-partial-register",
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
    assert len(rows) == 18
    assert rows[("mklq-cpu", "y-state", 1024)][
        "elapsed_seconds_median"] == 0.04462285348563455
    assert rows[("mklq-cpu", "cz-state", 1024)][
        "elapsed_seconds_median"] == 0.041044271012651734
    assert rows[("mklq-cpu", "qft-like-state", 1024)][
        "elapsed_seconds_median"] == 1.3367312084883451
    assert rows[("mklq-cpu", "seeded-clifford-state", 1024)][
        "elapsed_seconds_median"] == 0.13216056249802932
    assert rows[("mklq-cpu", "sample-partial-register", 65536)][
        "elapsed_seconds_median"] == 0.013367166495299898
    ratios = summary["comparison"]["clean_worktree_cross_target_ratio"]
    assert ratios["qpp_cpu_over_mklq_cpu_y_state_q20"] == pytest.approx(
        120.44143716035855)
    assert ratios["qpp_cpu_over_mklq_cpu_cz_state_q20"] == pytest.approx(
        121.46709470055829)
    assert ratios["qpp_cpu_over_mklq_cpu_qft_like_state_q20"] == pytest.approx(
        54.62876946186469)
    assert ratios[
        "qpp_cpu_over_mklq_cpu_seeded_clifford_state_q20"] == pytest.approx(
            97.56074012018514)
    assert ratios[
        "qpp_cpu_over_mklq_cpu_sample_partial_register_q20_65536_shots"
    ] == pytest.approx(120.60214077899167)


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


def test_mklq_benchmark_summary_records_metal_static_path_labels():
    repo_root = Path(__file__).resolve().parents[3]
    summary_path = (
        repo_root / "benchmarks" / "mklq" / "reports" /
        "local-metal-path-labels-q20-2026-06-22.summary.json")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert summary["schema_version"] == "mklq-benchmark-summary-v1"
    assert summary["evidence_kind"] == "local_tuning_evidence"
    assert summary["summary_id"] == "local-metal-path-labels-q20-2026-06-22"
    assert summary["git"]["commit"] == (
        "9a47901200f87ae9223bf232abb2b24767f8ac6f")
    assert summary["git"]["dirty"] is False
    assert summary["raw_results"][0]["path"] == (
        "benchmarks/mklq/results/"
        "local-metal-path-labels-state-q20-2026-06-22.json")
    assert summary["raw_results"][0]["sha256"] == (
        "5c44e7772c4842de44e68cbe138a760d6b301432faa533fdf16081f42489356b")
    assert summary["raw_results"][0]["status_rows"] == {"ok": 4}
    assert summary["raw_results"][1]["path"] == (
        "benchmarks/mklq/results/"
        "local-metal-path-labels-sampling-q20-2026-06-22.json")
    assert summary["raw_results"][1]["sha256"] == (
        "0087e0be2ca97238d6d9928f4c5ac974aeacbba96d1abb004ce4109bebd88dfd")
    assert summary["raw_results"][1]["status_rows"] == {"ok": 4}
    assert summary["config"]["targets"] == ["mklq-metal"]
    assert summary["config"]["cases"] == [
        "y-state",
        "cy-state",
        "qft-like-state",
        "seeded-clifford-state",
        "sample-full-register",
        "sample-partial-register",
    ]
    assert summary["config"]["qubits"] == [20]
    assert summary["config"]["shot_counts"] == [1024, 65536]

    rows = {
        (row["case"], row["shots"]): row
        for row in summary["rows"]
    }
    assert rows[("y-state", 1024)]["metal_path_label"] == (
        "mklq_metal_resident_single_gate_state_host_readback")
    assert rows[("cy-state", 1024)]["metal_path_label"] == (
        "mklq_metal_resident_controlled_gate_state_host_readback")
    assert rows[("qft-like-state", 1024)]["metal_path_label"] == (
        "mklq_metal_mixed_composite_state_host_readback")
    assert rows[("seeded-clifford-state", 1024)]["metal_path_label"] == (
        "mklq_metal_mixed_composite_state_host_readback")
    assert rows[("sample-full-register", 1024)]["metal_path_label"] == (
        "mklq_metal_mixed_sampling_host_counts")
    assert rows[("sample-partial-register", 65536)]["metal_path_label"] == (
        "mklq_metal_mixed_sampling_host_counts")
    for row in rows.values():
        assert row["metal_path_label_source"] == (
            "benchmark_harness_static_case_map")
        assert row["metal_runtime_counter"] is False
        assert row["metal_full_native"] is False
        assert "not a runtime counter" in row["metal_evidence_boundary"]
        assert row["elapsed_seconds_median"] > 0.0

    assert summary["interpretation"]["clean_worktree"] is True
    assert summary["interpretation"]["metal_path_labels_are_static_case_map"]
    assert summary["interpretation"]["metal_runtime_counter"] is False
    assert summary["interpretation"]["do_not_treat_as_clean_release_provenance"]


def test_mklq_benchmark_summary_records_metal_composite_evidence():
    repo_root = Path(__file__).resolve().parents[3]
    summary_path = (
        repo_root / "benchmarks" / "mklq" / "reports" /
        "local-metal-composite-mixed-path-q20-2026-06-21.summary.json")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert summary["schema_version"] == "mklq-benchmark-summary-v1"
    assert summary["evidence_kind"] == "local_tuning_evidence"
    assert summary["summary_id"] == (
        "local-metal-composite-mixed-path-q20-2026-06-21")
    assert summary["raw_results"][0]["path"] == (
        "benchmarks/mklq/results/"
        "local-metal-composite-mixed-path-q20-2026-06-21.json")
    assert summary["raw_results"][0]["status_rows"] == {"ok": 6}
    assert summary["machine"]["cpu_brand"] == "Apple M5"
    assert summary["config"]["targets"] == [
        "qpp-cpu",
        "mklq-cpu",
        "mklq-metal",
    ]
    assert summary["config"]["cases"] == [
        "qft-like-state",
        "seeded-clifford-state",
    ]
    assert summary["config"]["qubits"] == [20]
    assert summary["config"]["layers"] == 8

    rows = {
        (row["target"], row["case"]): row
        for row in summary["rows"]
    }
    assert rows[("mklq-metal", "qft-like-state")]["status"] == "ok"
    assert rows[("mklq-metal", "seeded-clifford-state")]["status"] == "ok"
    assert rows[("mklq-metal", "qft-like-state")][
        "elapsed_seconds_median"] > 0.0
    assert rows[("mklq-metal", "seeded-clifford-state")][
        "elapsed_seconds_median"] > 0.0
    ratios = summary["comparison"]["same_day_cross_target_ratio"]
    assert ratios["qpp_cpu_over_mklq_metal_qft_like_state_q20"] > 0.0
    assert ratios[
        "qpp_cpu_over_mklq_metal_seeded_clifford_state_q20"] > 0.0
    assert summary["interpretation"]["metal_path_scope"] == (
        "experimental mklq-metal mixed-path composite state-vector update "
        "followed by host readback for cudaq.get_state")
    assert summary["interpretation"]["do_not_treat_as_clean_release_provenance"]
    assert summary["interpretation"][
        "curated_path_labels_are_not_raw_benchmark_fields"] is True
