# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Define and statically estimate a concrete 15-to-1 T-state protocol."""

import cudaq.logical as ql


@ql.protocol(implements=ql.std.produce(ql.std.T_STATE))
def distill_15to1() -> ql.types.resource[ql.std.T_STATE]:
    """Turn 15 noisy T states into one postselected T state."""

    raw_states = ql.request_many(ql.std.RAW_T_STATE, count=15)
    output = ql.prepare_plus(
        ql.allocate_patch(
            ql.codes.BareQubit,
            region="t_state_factory",
        ))

    # Four raw states initialize the even-parity check rows. The fifth patch
    # is the odd row that survives as the output.
    checks = []
    for state in raw_states[:4]:
        output, check = ql.unpack_resource(
            state,
            like=output,
            encoding=ql.codes.BareQubit,
        )
        checks.append(check)

    rows = [*checks, output]
    for state, rotation in zip(
            raw_states[4:],
            ql.protocols.FIFTEEN_TO_ONE_ROTATION_STEPS,
    ):
        rows = list(rotation(*rows, state))

    # The positive-angle triorthogonal circuit produces T-dagger on the odd
    # row; S converts it to the canonical T|+> resource.
    rows[4] = ql.protocols.bare_s(rows[4])

    # Accept exactly when all four even rows measure +X.
    for check in rows[:4]:
        ql.postselect(
            ql.protocols.bare_measure_x(check),
            expected=False,
        )

    return ql.pack_resource(rows[4], kind=ql.std.T_STATE)


counts = ql.estimate(
    distill_15to1,
    tier=ql.estimate.Tier.STATIC,
)

assert counts.operation_counts["resource_request"] == 15
assert counts.operation_counts["resource_rotate_product"] == 11
assert counts.operation_counts["selection"] == 4
assert counts.success_count == 4
assert counts.operation_counts["pack_resource"] == 1

print("15-to-1 static resource estimate:")
print(f"  raw T-state requests: {counts.resource_requests['raw_t_state']}")
print(
    f"  resource rotations: {counts.operation_counts['resource_rotate_product']}"
)
print(f"  postselection checks: {counts.success_count}")
print(f"  peak live patches: {counts.patches_peak}")
