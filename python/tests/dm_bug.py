import cudaq

# Unable to reproduce the "bug" as follows from the builder:

def state_vector():
  cudaq.set_target("qpp-cpu")
  kernel = cudaq.make_kernel()
  q0 = kernel.qalloc()
  kernel.x(q0)

  # Allocate another qubit.
  q1 = kernel.qalloc()

  # Should measure |10>
  result = cudaq.sample(kernel)
  print(result)

  print(cudaq.get_state(kernel))


def density_matrix():
  cudaq.set_target("density-matrix-cpu")
  kernel = cudaq.make_kernel()
  q0 = kernel.qalloc()
  kernel.x(q0)

  # Allocate another qubit.
  q1 = kernel.qalloc()

  print(kernel)

  # Should measure |10>
  result = cudaq.sample(kernel)
  print(result)

  print(cudaq.get_state(kernel))


state_vector()
density_matrix()