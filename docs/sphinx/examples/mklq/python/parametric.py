#!/usr/bin/env python3
"""Sample a deterministic parameterized-gate fixture on MKL-Q targets."""

import argparse

import cudaq


@cudaq.kernel
def parametric():
    q = cudaq.qvector(3)
    ry(3.141592653589793, q[0])
    rx(3.141592653589793, q[1])
    rz(1.5707963267948966, q[2])
    x.ctrl(q[0], q[2])
    mz(q)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the MKL-Q parameterized-gate example.")
    parser.add_argument("--target",
                        action="append",
                        choices=("mklq-cpu", "mklq-metal"),
                        help="Target to run. Repeat to run multiple targets.")
    parser.add_argument("--shots",
                        type=int,
                        default=100,
                        help="Number of samples to collect.")
    return parser.parse_args()


def main():
    args = parse_args()
    targets = args.target or ["mklq-cpu", "mklq-metal"]
    for target in targets:
        cudaq.set_target(target)
        counts = cudaq.sample(parametric, shots_count=args.shots)
        print(f"{target}: {counts}")


if __name__ == "__main__":
    main()
