#!/usr/bin/env python3
"""Sample a three-qubit GHZ state on MKL-Q targets."""

import argparse

import cudaq


@cudaq.kernel
def ghz():
    q = cudaq.qvector(3)
    h(q[0])
    x.ctrl(q[0], q[1])
    x.ctrl(q[1], q[2])
    mz(q)


def parse_args():
    parser = argparse.ArgumentParser(description="Run the MKL-Q GHZ example.")
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
        counts = cudaq.sample(ghz, shots_count=args.shots)
        print(f"{target}: {counts}")


if __name__ == "__main__":
    main()
