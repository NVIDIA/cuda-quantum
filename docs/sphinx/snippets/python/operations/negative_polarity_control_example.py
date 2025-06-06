import cudaq

# [Begin Negative Polarity Python]
@cudaq.kernel
def kernel_neg_polarity_py():
    q, r = cudaq.qubit(), cudaq.qubit()

    // To demonstrate the effect of ~q:
    // If q is |0> (default), ~q is true (control on |0>), X is applied to r. r becomes |1>.
    // If q is |1> (after an X(q)), ~q is false, X is not applied to r. r remains |0>.

    // Let's test the case where q is |1> initially.
    x(q) # q is now |1>
    x.ctrl(~q, r) # ~q is false (control on |0>), so X is NOT applied to r. r remains |0>.
    mz(q,r) # Expect bitstring "10" (q=1, r=0)

    print("Python: Negative polarity kernel executed.")
# [End Negative Polarity Python]

# [Begin Negative Polarity Python Execution]
if __name__ == "__main__":
    print("Python Negative Polarity Control Example:")
    counts = cudaq.sample(kernel_neg_polarity_py)
    counts.dump()
    # Expected output for the case where q starts as |1|:
    # { 10:[shots] }
# [End Negative Polarity Python Execution]