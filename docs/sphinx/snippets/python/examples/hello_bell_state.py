import cudaq

# [Begin Bell State Python]
@cudaq.kernel()
def bell(num_iters : int) -> int:
    q = cudaq.qvector(2)
    nCorrect = 0
    for i in range(num_iters):
        h(q[0])
        x.ctrl(q[0], q[1])
        results = mz(q)
        if results[0] == results[1]:
           nCorrect = nCorrect + 1
        
        reset(q) # reset the whole qvector
    return nCorrect

counts = bell(num_iters=100) # Pass as keyword arg for clarity
print(f'N Correct = {counts}')
assert counts == 100
# [End Bell State Python]

if __name__ == "__main__":
    pass # Logic is at top level