# MKL-Q Metal Runtime Counter Summary

This file is generated from bounded `.counter.json` reports under `benchmarks/mklq/reports/`.

Caveat: this is runtime counter evidence from selected build-tree ctest cases. It is not release sign-off, not a benchmark result, and not proof that every operation stayed on Metal.

## Aggregate

| Field | Value |
| --- | --- |
| `status` | passed |
| `report_count` | 1 |
| `expected` | 19 |
| `selected` | 19 |
| `missing` | 0 |
| `passed` | 19 |
| `failed` | 0 |

## Evidence Boundary

| Boundary | Value |
| --- | --- |
| `runtime_counter_evidence` | True |
| `release_signoff` | False |
| `all_metal_execution_proof` | False |
| `raw_logs_truncated` | True |

## Counter Coverage Categories

| Category | Passed | Failed | Other | Description |
| --- | ---: | ---: | ---: | --- |
| fallback_boundary | 1 | 0 | 0 | Unsupported-gate fallback and reupload boundary tests |
| measurement_reset | 3 | 0 | 0 | Measurement, collapse, and reset counter tests |
| probability_sampling | 5 | 0 | 0 | Resident probability fill and sampling counter tests |
| resident_gate | 10 | 0 | 0 | Resident Metal gate/update counter tests |

## Counter Tests

| Category | Test |
| --- | --- |
| fallback_boundary | `mklq_metal_MKLQMetalTester.SimulatorReuploadsResidentStateAfterUnsupportedGateFallback` |
| measurement_reset | `mklq_metal_MKLQMetalTester.MetalRuntimeComputesAndCollapsesResidentQubitProbability` |
| measurement_reset | `mklq_metal_MKLQMetalTester.SimulatorMeasuresAndResetsResidentStateWithoutReadback` |
| measurement_reset | `mklq_metal_MKLQMetalTester.SimulatorResetsResidentNonzeroTargetWithoutReadback` |
| probability_sampling | `mklq_metal_MKLQMetalTester.MetalRuntimeFillsResidentProbabilitiesWithoutStateReadback` |
| probability_sampling | `mklq_metal_MKLQMetalTester.SimulatorSamplesDenseFullRegisterThroughMetalProbabilityFill` |
| probability_sampling | `mklq_metal_MKLQMetalTester.SimulatorSamplesLargeResidentPartialRegisterThroughFullProbability` |
| probability_sampling | `mklq_metal_MKLQMetalTester.SimulatorSamplesResidentDenseStateWithoutReadback` |
| probability_sampling | `mklq_metal_MKLQMetalTester.SimulatorSamplesSmallResidentPartialRegisterThroughMarginalProbability` |
| resident_gate | `mklq_metal_MKLQMetalTester.MetalRuntimeKeepsResidentStateAcrossGateSequence` |
| resident_gate | `mklq_metal_MKLQMetalTester.SimulatorKeepsBuiltInPhaseFamilyResidentUntilReadback` |
| resident_gate | `mklq_metal_MKLQMetalTester.SimulatorKeepsBuiltInRxAndControlledRxResidentUntilReadback` |
| resident_gate | `mklq_metal_MKLQMetalTester.SimulatorKeepsBuiltInRyAndControlledRyResidentUntilReadback` |
| resident_gate | `mklq_metal_MKLQMetalTester.SimulatorKeepsBuiltInRzAndControlledRzResidentUntilReadback` |
| resident_gate | `mklq_metal_MKLQMetalTester.SimulatorKeepsBuiltInYAndControlledYResidentUntilReadback` |
| resident_gate | `mklq_metal_MKLQMetalTester.SimulatorKeepsMultiControlSingleQubitResidentUntilReadback` |
| resident_gate | `mklq_metal_MKLQMetalTester.SimulatorKeepsSupportedGateSequenceResidentUntilReadback` |
| resident_gate | `mklq_metal_MKLQMetalTester.SimulatorKeepsThreeQubitGateResidentUntilReadback` |
| resident_gate | `mklq_metal_MKLQMetalTester.SimulatorKeepsYAndControlledYResidentUntilReadback` |

## Reports

| Report | Created | Status | Expected | Selected | Missing | Passed | Failed |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| benchmarks/mklq/reports/local-metal-runtime-counter-probe-2026-06-22.counter.json | 2026-06-22T07:25:49.516754+00:00 | passed | 19 | 19 | 0 | 19 | 0 |

Regenerate with:

```bash
python3 benchmarks/mklq/summarize_metal_runtime_counters.py \
  --reports benchmarks/mklq/reports \
  --output docs/mklq/metal-runtime-counters.md
```
