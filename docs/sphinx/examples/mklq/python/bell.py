#!/usr/bin/env python3
"""Sample a Bell state on MKL-Q targets."""

import argparse

import cudaq


@cudaq.kernel
def bell():
    q = cudaq.qvector(2)
    h(q[0])
    x.ctrl(q[0], q[1])
    mz(q)


def parse_args():
    parser = argparse.ArgumentParser(description="Run the MKL-Q Bell example.")
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
        counts = cudaq.sample(bell, shots_count=args.shots)
        print(f"{target}: {counts}")


if __name__ == "__main__":
    main()
