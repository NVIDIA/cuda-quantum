// Compile and run with:
// ```
// mpic++ mpi_cuda_check.cpp -o check.x && mpiexec -np 1 ./check.x
// ```

#include "mpi.h"
#if __has_include("mpi-ext.h")
#include "mpi-ext.h"
#endif
#include <stdio.h>

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  int exit_code;
  if (MPIX_Query_cuda_support()) {
    printf("CUDA-aware MPI installation.\n");
    exit_code = 0;
  } else {
    printf("Missing CUDA support.\n");
    exit_code = 1;
  }
  MPI_Finalize();
  return exit_code;
}