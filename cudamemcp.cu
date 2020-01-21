#include <stdio.h>

__global__ void kernel(int* a_d, int* b_d, int* c_d){
  *c_d = *a_d + *b_d;
  return;
}

int main(){
  int a = 1, b = 2;
  int *a_d, *b_d, *c_d;
  cudaMalloc((void**) &a_d, sizeof(int));
  cudaMalloc((void**) &b_d, sizeof(int));
  cudaMalloc((void**) &c_d, sizeof(int));

  cudaMemcpy(a_d, &a, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(b_d, &b, sizeof(int), cudaMemcpyHostToDevice);
  int c;

  kernel<<<1, 1>>>(a_d, b_d, c_d);
  cudaMemcpy(&c, c_d, sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree((void**) a_d);
  cudaFree((void**) b_d);
  cudaFree((void**) c_d);
  printf("%d\n", c);
}
