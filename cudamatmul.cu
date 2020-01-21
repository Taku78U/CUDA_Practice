#include <stdio.h>

// pitch: the number of cols
// size : the number of rows
__global__ void matmul_kernel(
    const float* const mat1, const float* const mat2, float* const mat3,
    const size_t pitch1, const size_t pitch2, const size_t pitch3,
    const size_t size1, const size_t size2, const size_t size3
                              ){
  for(size_t i = 0; i < size1; ++i){
    for(size_t j = 0; j < size3; ++j){
      for(size_t k = 0; k < size2; ++k){
        mat3[i*pitch3+j] += mat1[i*pitch1+k] * mat2[k*pitch2+j];
      }
    }
  }
  return;
}

void print_array(float* arr, size_t pitch, size_t size){
  for(size_t i = 0; i < size; ++i){
    for(size_t j = 0; j < pitch; ++j){
      printf("%f ", arr[i * size + j]);
    }
    printf("\n");
  }
}

int main(){
  const size_t pitch1 = 3;
  const size_t pitch2 = 3;

  const size_t size1 = 3;
  const size_t size2 = pitch1;

  const size_t pitch3 = pitch2;
  const size_t size3 = size1;

  const size_t n1 = pitch1 * size1;
  const size_t n2 = pitch2 * size2;
  const size_t n3 = pitch3 * size3;

  float a[9] = {5, 2, 1, 5, 2, 1, 5, 2, 1};
  float b[9] = {5, 2, 1, 5, 2, 1, 5, 2, 1};
  float c[9] = {0};

  float *a_d, *b_d, *c_d;

  cudaMalloc((void**) &a_d, n1 * sizeof(float));
  cudaMalloc((void**) &b_d, n2 * sizeof(float));
  cudaMalloc((void**) &c_d, n3 * sizeof(float));

  cudaMemcpy(a_d, a, n1 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(b_d, b, n2 * sizeof(float), cudaMemcpyHostToDevice);

  matmul_kernel<<<3, 3>>>(a_d, b_d, c_d, pitch1, pitch2, pitch3, size1, size2, size3);

  cudaMemcpy(c, c_d, n3 * sizeof(float), cudaMemcpyDeviceToHost);

  print_array(c, pitch3, size3);
  
}
