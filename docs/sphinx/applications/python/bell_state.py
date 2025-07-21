import cudaq


@cudaq.kernel()
def bell(num_iters: int) -> int:
    q = cudaq.qvector(2)
    nCorrect = 0
    for i in range(num_iters):
        h(q[0])
        x.ctrl(q[0], q[1])
        results = mz(q)
        if results[0] == results[1]:
            nCorrect = nCorrect + 1

        reset(q)
    return nCorrect


counts = bell(100)
print(f"N Correct = {counts}")
assert counts == 100
