#!/usr/bin/env python3
"""Sample a deterministic Clifford-chain fixture on MKL-Q targets."""

import argparse

import cudaq


@cudaq.kernel
def clifford_chain():
    q = cudaq.qvector(4)
    x(q[0])
    x(q[3])
    swap(q[0], q[2])
    h(q[1])
    z(q[1])
    h(q[1])
    x.ctrl(q[2], q[0])
    cz(q[0], q[3])
    s(q[2])
    sdg(q[2])
    mz(q)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the MKL-Q Clifford-chain example.")
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
        counts = cudaq.sample(clifford_chain, shots_count=args.shots)
        print(f"{target}: {counts}")


if __name__ == "__main__":
    main()
