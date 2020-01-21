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

__global__ void matmul_parallel_kernel(
    const float* const mat1, const float* const mat2, float* const mat3,
    const size_t pitch1, const size_t pitch2, const size_t pitch3,
    const size_t size1, const size_t size2, const size_t size3
                              ){
  // blockIdx.y: Grid's y-axis (= Mat1 and Mat3 row-axis | bv) index
  // blockIdx.x: Grid's x-axis (= Mat2 and Mat3 col-axis | bh) index
  const size_t i = blockIdx.y * blockDim.y + threadIdx.y;
  const size_t j = blockIdx.x * blockDim.x + threadIdx.x;

  if((i < size1) && (j < size3)){
    for(size_t k = 0; k < size2; ++k){
      mat3[i*pitch3+j] += mat1[i*pitch1+k] * mat2[k*pitch2+j];
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
  const int threadsPerBlock = 8; 
  
  const size_t pitch1 = 3;
  const size_t pitch2 = 3;

  const size_t size1 = 3;
  const size_t size2 = pitch1;

  const size_t pitch3 = pitch2;
  const size_t size3 =  pitch2;

  const size_t n1 = pitch1 * size1;
  const size_t n2 = pitch2 * size2;
  const size_t n3 = pitch3 * size3;

  float a[9] = {5, 2, 1, 5, 2, 1, 5, 2, 1};
  float b[9] = {5, 2, 1, 5, 2, 1, 5, 2, 1};
  float c[9] = {0};

  float *a_d, *b_d, *c_d;

  size_t bv = (size1 + threadsPerBlock - 1) / threadsPerBlock;
  size_t bh = (size3 + threadsPerBlock - 1) / threadsPerBlock;

  // Grid x: bh is: Matrix2 and Matrix3 col-axis
  //      y: bv is: Matrix1 and Matrix3 row-axis
  // Block x, y: Not particularly relevant to the matrix axis
  dim3 block(threadsPerBlock, threadsPerBlock);
  dim3 grid(bh, bv);

  cudaMalloc((void**) &a_d, n1 * sizeof(float));
  cudaMalloc((void**) &b_d, n2 * sizeof(float));
  cudaMalloc((void**) &c_d, n3 * sizeof(float));

  cudaMemcpy(a_d, a, n1 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(b_d, b, n2 * sizeof(float), cudaMemcpyHostToDevice);

  matmul_parallel_kernel<<<grid, block>>>(a_d, b_d, c_d, pitch1, pitch2, pitch3, size1, size2, size3);

  cudaMemcpy(c, c_d, n3 * sizeof(float), cudaMemcpyDeviceToHost);

  print_array(c, pitch3, size3);

  return 0;
  
}
