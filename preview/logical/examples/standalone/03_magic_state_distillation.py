# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Define and statically estimate a 15-to-1 T-state protocol."""

# %%
# Import the standalone CUDA-Q Logical protocol API.
import cudaq.logical as cql


# %%
# Define a protocol that consumes 15 raw states and produces one T state.
@cql.protocol(implements=cql.logical.produce(cql.logical.T_STATE))
def distill_15to1() -> cql.types.resource[cql.logical.T_STATE]:
    raw_states = cql.request_many(
        cql.logical.RAW_T_STATE,
        count=15,
    )
    output = cql.prepare_plus(
        cql.allocate_patch(cql.codes.BareQubit, region="t_state_factory"))

    checks = []
    for state in raw_states[:4]:
        output, check = cql.unpack_resource(
            state,
            like=output,
            encoding=cql.codes.BareQubit,
        )
        checks.append(check)

    rows = [*checks, output]
    for state, rotation in zip(raw_states[4:],
                               cql.protocols.FIFTEEN_TO_ONE_ROTATION_STEPS):
        rows = list(rotation(*rows, state))
    rows[4] = cql.protocols.bare_s(rows[4])

    for check in rows[:4]:
        cql.postselect(
            cql.protocols.bare_measure_x(check),
            expected=False,
        )
    return cql.pack_resource(
        rows[4],
        kind=cql.logical.T_STATE,
    )


# %%
# Estimate the protocol directly at its encoded static-resource layer.
resources = cql.estimate(distill_15to1, tier=cql.estimate.Tier.STATIC)

assert resources.operation_counts["resource_request"] == 15
assert resources.operation_counts["resource_rotate_product"] == 11
assert resources.operation_counts["selection"] == 4
assert resources.operation_counts["pack_resource"] == 1

print("15-to-1 static resource estimate:")
print(f"  raw T-state requests: "
      f"{resources.operation_counts['resource_request']}")
print(f"  peak live patches: {resources.patches_peak}")
