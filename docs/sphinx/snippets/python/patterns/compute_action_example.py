import cudaq
import numpy as np

# This kernel will contain the parts shown in the RST
@cudaq.kernel
def kernel_compute_action_py():
    q = cudaq.qvector(1) # q needs to be defined for the functions/lambda

    # [Begin Compute Action Python Snippet]
    def computeF():
       h(q[0])
       s(q[0])

    def actionF():
       x(q[0])

    # Can take user-defined functions
    cudaq.compute_action(computeF, actionF)

    # Can take Pythonic CUDA-Q lambda kernels (callables)
    # Ensure q is accessible in the lambda's scope
    computeL = lambda : (h(q[0]), x(q[0]), ry(-np.pi, q[0]))
    # actionF can be reused, or define a new one for the lambda example
    cudaq.compute_action(computeL, actionF)
    #  [End Compute Action Python Snippet]

    mz(q)

if __name__ == "__main__":
    print("Python Compute-Action Example:")
    counts = cudaq.sample(kernel_compute_action_py)
    counts.dump()
    # The final state is a result of two sequential compute_action blocks.
    # The first compute_action(computeF, actionF) results in S H X H S_adj.
    # The second compute_action(computeL, actionF) applies U_L V_actionF U_L_adj
    # where U_L is Ry(-pi) X H.
    # The exact final distribution can be complex but the test ensures it runs.