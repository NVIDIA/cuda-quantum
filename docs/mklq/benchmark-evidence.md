# MKL-Q Benchmark Evidence

This file is generated from sanitized benchmark summaries under `benchmarks/mklq/reports/`.

Caveat: these entries are local benchmark evidence from development or release-prep runs. Interpret each entry through its `evidence_kind` and `interpretation` fields; none is a cross-machine performance certification.

## Evidence Inventory

| Summary ID | Kind | Machine | Targets | Cases | Qubits | Run shape | Rows | Raw evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| local-clean-cpu-q20-2026-06-21 | clean_local_benchmark_evidence | Apple M5, 10 logical cores, 16 GiB RAM, macOS 26.5.1 | qpp-cpu, mklq-cpu | y-state, cy-state, cz-state, qft-like-state, seeded-clifford-state, sample-full-register, sample-partial-register | 20 | shot_counts=1024, 65536; repeats=2; warmups=1; layers=8; isolate_rows=true | ok=18 | benchmarks/mklq/results/local-clean-cpu-gate-y-cy-cz-q20-2026-06-21.json sha256=2b438094b63b; benchmarks/mklq/results/local-clean-cpu-composite-qft-like-seeded-clifford-q20-2026-06-21.json sha256=b07b3ba92b83; benchmarks/mklq/results/local-clean-cpu-sampling-q20-2026-06-21.json sha256=167b5c4adef8 |
| local-counts-only-sampling-shot-scaling-q20-2026-06-19 | local_tuning_evidence | Apple M5, 10 logical cores, 16 GiB RAM, macOS 26.5.1 | qpp-cpu, mklq-cpu, mklq-metal | sample-full-register, sample-partial-register | 20 | shot_counts=256, 1024, 8192, 65536; repeats=2; warmups=1; layers=8; isolate_rows=true | ok=24 | benchmarks/mklq/results/local-counts-only-sampling-shot-scaling-q20-2026-06-19.json sha256=ef9846673b46 |
| local-current-sampling-fullprob-gated-q20-2026-06-19 | local_tuning_evidence | Apple M5, 10 logical cores, 16 GiB RAM, macOS 26.5.1 | qpp-cpu, mklq-cpu, mklq-metal | sample-full-register, sample-partial-register | 20 | shots=1024; repeats=2; warmups=1; layers=4; isolate_rows=true | ok=6 | benchmarks/mklq/results/local-current-sampling-fullprob-gated-q20-2026-06-19.json sha256=8ca6a4f7a7ae; benchmarks/mklq/results/local-current-sampling-shot-scaling-q20-2026-06-19.json sha256=9c15c0c1d566 |
| local-metal-composite-mixed-path-q20-2026-06-21 | local_tuning_evidence | Apple M5, 10 logical cores, 16 GiB RAM, macOS 26.5.1 | qpp-cpu, mklq-cpu, mklq-metal | qft-like-state, seeded-clifford-state | 20 | shot_counts=1024; repeats=2; warmups=1; layers=8; isolate_rows=true | ok=6 | benchmarks/mklq/results/local-metal-composite-mixed-path-q20-2026-06-21.json sha256=ef58b5922221 |
| local-metal-path-labels-q20-2026-06-22 | local_tuning_evidence | Apple M5, 10 logical cores, 16 GiB RAM, macOS 26.5.1 | mklq-metal | y-state, cy-state, qft-like-state, seeded-clifford-state, sample-full-register, sample-partial-register | 20 | shot_counts=1024, 65536; repeats=2; warmups=1; layers=8; isolate_rows=true | ok=8 | benchmarks/mklq/results/local-metal-path-labels-state-q20-2026-06-22.json sha256=5c44e7772c48; benchmarks/mklq/results/local-metal-path-labels-sampling-q20-2026-06-22.json sha256=0087e0be2ca9 |
| local-metal-three-qubit-resident-q20-2026-06-22 | local_tuning_evidence | Apple M5, 10 logical cores, 16 GiB RAM, macOS 26.5.1 | qpp-cpu, mklq-cpu, mklq-metal | three-qubit-state | 20 | shot_counts=1024; repeats=2; warmups=1; layers=8; isolate_rows=true | ok=3 | benchmarks/mklq/results/local-metal-three-qubit-resident-q20-2026-06-22.json sha256=daed4c1deb2d |
| local-metal-y-cy-resident-isolated-q20-2026-06-19 | local_tuning_evidence | Apple M5, 10 logical cores, 16 GiB RAM, macOS 26.5.1 | qpp-cpu, mklq-cpu, mklq-metal | y-state, cy-state | 20 | shots=1024; repeats=2; warmups=1; layers=8; isolate_rows=true | ok=6 | benchmarks/mklq/results/local-metal-y-cy-resident-isolated-q20-2026-06-19.json sha256=84891e8f907c |
| local-y-cy-fastpath-isolated-q20-2026-06-19 | local_tuning_evidence | Apple M5, 10 logical cores, 16 GiB RAM, macOS 26.5.1 | qpp-cpu, mklq-cpu | y-state, cy-state | 20 | shots=1024; repeats=2; warmups=1; layers=8; isolate_rows=true | ok=4 | benchmarks/mklq/results/local-y-cy-fastpath-isolated-q20-2026-06-19.json sha256=93bce3b77fcc |

## Comparison Signals

The values below are copied from each summary's bounded `comparison` object. Keep their original local-run context when citing them.

| Summary ID | Metric | Value |
| --- | --- | --- |
| local-clean-cpu-q20-2026-06-21 | `clean_worktree_cross_target_ratio.qpp_cpu_over_mklq_cpu_cy_state_q20` | 99.08x |
| local-clean-cpu-q20-2026-06-21 | `clean_worktree_cross_target_ratio.qpp_cpu_over_mklq_cpu_cz_state_q20` | 121.47x |
| local-clean-cpu-q20-2026-06-21 | `clean_worktree_cross_target_ratio.qpp_cpu_over_mklq_cpu_qft_like_state_q20` | 54.63x |
| local-clean-cpu-q20-2026-06-21 | `clean_worktree_cross_target_ratio.qpp_cpu_over_mklq_cpu_sample_full_register_q20_1024_shots` | 139.93x |
| local-clean-cpu-q20-2026-06-21 | `clean_worktree_cross_target_ratio.qpp_cpu_over_mklq_cpu_sample_full_register_q20_65536_shots` | 106.66x |
| local-clean-cpu-q20-2026-06-21 | `clean_worktree_cross_target_ratio.qpp_cpu_over_mklq_cpu_sample_partial_register_q20_1024_shots` | 166.86x |
| local-clean-cpu-q20-2026-06-21 | `clean_worktree_cross_target_ratio.qpp_cpu_over_mklq_cpu_sample_partial_register_q20_65536_shots` | 120.60x |
| local-clean-cpu-q20-2026-06-21 | `clean_worktree_cross_target_ratio.qpp_cpu_over_mklq_cpu_seeded_clifford_state_q20` | 97.56x |
| local-clean-cpu-q20-2026-06-21 | `clean_worktree_cross_target_ratio.qpp_cpu_over_mklq_cpu_y_state_q20` | 120.44x |
| local-clean-cpu-q20-2026-06-21 | `mklq_cpu_elapsed_seconds_median.cy_state_q20` | 0.0560914 s |
| local-clean-cpu-q20-2026-06-21 | `mklq_cpu_elapsed_seconds_median.cz_state_q20` | 0.0410443 s |
| local-clean-cpu-q20-2026-06-21 | `mklq_cpu_elapsed_seconds_median.qft_like_state_q20` | 1.33673 s |
| local-clean-cpu-q20-2026-06-21 | `mklq_cpu_elapsed_seconds_median.sample_full_register_q20_1024_shots` | 0.0128129 s |
| local-clean-cpu-q20-2026-06-21 | `mklq_cpu_elapsed_seconds_median.sample_full_register_q20_65536_shots` | 0.0157315 s |
| local-clean-cpu-q20-2026-06-21 | `mklq_cpu_elapsed_seconds_median.sample_partial_register_q20_1024_shots` | 0.010787 s |
| local-clean-cpu-q20-2026-06-21 | `mklq_cpu_elapsed_seconds_median.sample_partial_register_q20_65536_shots` | 0.0133672 s |
| local-clean-cpu-q20-2026-06-21 | `mklq_cpu_elapsed_seconds_median.seeded_clifford_state_q20` | 0.132161 s |
| local-clean-cpu-q20-2026-06-21 | `mklq_cpu_elapsed_seconds_median.y_state_q20` | 0.0446229 s |
| local-counts-only-sampling-shot-scaling-q20-2026-06-19 | `q20_65536_shot_elapsed_ratio_qpp_cpu_over_mklq_cpu.sample_full_register` | 58.54x |
| local-counts-only-sampling-shot-scaling-q20-2026-06-19 | `q20_65536_shot_elapsed_ratio_qpp_cpu_over_mklq_cpu.sample_partial_register` | 83.64x |
| local-counts-only-sampling-shot-scaling-q20-2026-06-19 | `q20_65536_shot_elapsed_ratio_qpp_cpu_over_mklq_metal.sample_full_register` | 27.94x |
| local-counts-only-sampling-shot-scaling-q20-2026-06-19 | `q20_65536_shot_elapsed_ratio_qpp_cpu_over_mklq_metal.sample_partial_register` | 38.01x |
| local-current-sampling-fullprob-gated-q20-2026-06-19 | `pre_gate_probe.mklq_metal_sample_full_register_q20_seconds` | 0.0313161 s |
| local-current-sampling-fullprob-gated-q20-2026-06-19 | `pre_gate_probe.mklq_metal_sample_partial_register_q20_seconds` | 0.255697 s |
| local-current-sampling-fullprob-gated-q20-2026-06-19 | `pre_gate_probe.repeats` | 1 |
| local-current-sampling-fullprob-gated-q20-2026-06-19 | `pre_gate_probe.shots` | 1024 |
| local-current-sampling-fullprob-gated-q20-2026-06-19 | `same_day_tuning_ratio.mklq_metal_sample_partial_register_pre_gate_over_cost_gate` | 11.62x |
| local-metal-composite-mixed-path-q20-2026-06-21 | `mklq_metal_elapsed_seconds_median.qft_like_state_q20` | 1.21681 s |
| local-metal-composite-mixed-path-q20-2026-06-21 | `mklq_metal_elapsed_seconds_median.seeded_clifford_state_q20` | 0.189254 s |
| local-metal-composite-mixed-path-q20-2026-06-21 | `same_day_cross_target_ratio.qpp_cpu_over_mklq_metal_qft_like_state_q20` | 56.00x |
| local-metal-composite-mixed-path-q20-2026-06-21 | `same_day_cross_target_ratio.qpp_cpu_over_mklq_metal_seeded_clifford_state_q20` | 70.50x |
| local-metal-path-labels-q20-2026-06-22 | `mklq_metal_elapsed_seconds_median.cy_state_q20` | 0.195022 s |
| local-metal-path-labels-q20-2026-06-22 | `mklq_metal_elapsed_seconds_median.qft_like_state_q20` | 1.33702 s |
| local-metal-path-labels-q20-2026-06-22 | `mklq_metal_elapsed_seconds_median.sample_full_register_q20_1024_shots` | 0.0299139 s |
| local-metal-path-labels-q20-2026-06-22 | `mklq_metal_elapsed_seconds_median.sample_full_register_q20_65536_shots` | 0.0317822 s |
| local-metal-path-labels-q20-2026-06-22 | `mklq_metal_elapsed_seconds_median.sample_partial_register_q20_1024_shots` | 0.059596 s |
| local-metal-path-labels-q20-2026-06-22 | `mklq_metal_elapsed_seconds_median.sample_partial_register_q20_65536_shots` | 0.0324173 s |
| local-metal-path-labels-q20-2026-06-22 | `mklq_metal_elapsed_seconds_median.seeded_clifford_state_q20` | 0.164919 s |
| local-metal-path-labels-q20-2026-06-22 | `mklq_metal_elapsed_seconds_median.y_state_q20` | 0.14844 s |
| local-metal-three-qubit-resident-q20-2026-06-22 | `mklq_metal_elapsed_seconds_median.three_qubit_state_q20` | 0.155606 s |
| local-metal-three-qubit-resident-q20-2026-06-22 | `same_day_cross_target_ratio.qpp_cpu_over_mklq_metal_three_qubit_state_q20` | 54.67x |
| local-metal-y-cy-resident-isolated-q20-2026-06-19 | `same_day_cross_target_ratio.mklq_metal_over_mklq_cpu_cy_state_q20` | 1.74x |
| local-metal-y-cy-resident-isolated-q20-2026-06-19 | `same_day_cross_target_ratio.mklq_metal_over_mklq_cpu_y_state_q20` | 2.13x |
| local-metal-y-cy-resident-isolated-q20-2026-06-19 | `same_day_cross_target_ratio.qpp_cpu_over_mklq_metal_cy_state_q20` | 53.20x |
| local-metal-y-cy-resident-isolated-q20-2026-06-19 | `same_day_cross_target_ratio.qpp_cpu_over_mklq_metal_y_state_q20` | 56.16x |
| local-y-cy-fastpath-isolated-q20-2026-06-19 | `same_day_cross_target_ratio.qpp_cpu_over_mklq_cpu_cy_state_q20` | 103.85x |
| local-y-cy-fastpath-isolated-q20-2026-06-19 | `same_day_cross_target_ratio.qpp_cpu_over_mklq_cpu_y_state_q20` | 167.38x |

## Metal Path Labels

The rows below expose static benchmark case-map labels for `mklq-metal`. They are not runtime counters or proof that every operation stayed on Metal.

| Summary ID | Case | Shots | Label | Scope | Source |
| --- | --- | --- | --- | --- | --- |
| local-metal-path-labels-q20-2026-06-22 | y-state | 1024 | `mklq_metal_resident_single_gate_state_host_readback` | resident fp32 Metal single-target gate update followed by host readback for cudaq.get_state | benchmark_harness_static_case_map |
| local-metal-path-labels-q20-2026-06-22 | cy-state | 1024 | `mklq_metal_resident_controlled_gate_state_host_readback` | resident fp32 Metal controlled gate update followed by host readback for cudaq.get_state | benchmark_harness_static_case_map |
| local-metal-path-labels-q20-2026-06-22 | qft-like-state | 1024 | `mklq_metal_mixed_composite_state_host_readback` | experimental mklq-metal mixed-path composite state-vector update followed by host readback for cudaq.get_state | benchmark_harness_static_case_map |
| local-metal-path-labels-q20-2026-06-22 | seeded-clifford-state | 1024 | `mklq_metal_mixed_composite_state_host_readback` | experimental mklq-metal mixed-path composite state-vector update followed by host readback for cudaq.get_state | benchmark_harness_static_case_map |
| local-metal-path-labels-q20-2026-06-22 | sample-full-register | 1024 | `mklq_metal_mixed_sampling_host_counts` | mixed-path Metal probability fill with host-side sample draw/count accumulation | benchmark_harness_static_case_map |
| local-metal-path-labels-q20-2026-06-22 | sample-full-register | 65536 | `mklq_metal_mixed_sampling_host_counts` | mixed-path Metal probability fill with host-side sample draw/count accumulation | benchmark_harness_static_case_map |
| local-metal-path-labels-q20-2026-06-22 | sample-partial-register | 1024 | `mklq_metal_mixed_sampling_host_counts` | mixed-path Metal probability fill with host-side sample draw/count accumulation | benchmark_harness_static_case_map |
| local-metal-path-labels-q20-2026-06-22 | sample-partial-register | 65536 | `mklq_metal_mixed_sampling_host_counts` | mixed-path Metal probability fill with host-side sample draw/count accumulation | benchmark_harness_static_case_map |
| local-metal-three-qubit-resident-q20-2026-06-22 | three-qubit-state | 1024 | `mklq_metal_resident_three_gate_state_host_readback` | resident fp32 Metal three-target gate update followed by host readback for cudaq.get_state | benchmark_harness_static_case_map |

Regenerate with:

```bash
python3 benchmarks/mklq/summarize_reports.py \
  --reports benchmarks/mklq/reports \
  --format markdown \
  --output docs/mklq/benchmark-evidence.md
```
