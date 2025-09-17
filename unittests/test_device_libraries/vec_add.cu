#include <cuda.h>
#include <stdio.h>

extern "C" {
__global__ void vectorAdd(float *a, float *b, float *c, int n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i == 0)
    printf("Calling the function %d!\n", n);
  if (i < n) {
    c[i] = a[i] + b[i];
  }
}
}
