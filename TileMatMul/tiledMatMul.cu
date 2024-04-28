#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <math.h>
#include <cstdlib>
#include <cassert>
#include <vector>
#include <functional>

using namespace std;
using std::cout;
using std::generate;
using std::vector;

const int N = 1 << 10;
const int SHMEM_SIZE = 1 << 10;

__global__
void matrixMul(int *a, int *b, int *c) {
    // Allocate shared memory
    __shared__ int A[SHMEM_SIZE];
    __shared__ int B[SHMEM_SIZE];

    // Calculate each thread's global row and column
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    // Extract some builtin values to simplify code
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int dim = blockDim.x; // symmetrical in this case

    int temp = 0;
    // Move the tile across the length of the matrix
    for (int i = 0; i < (N / dim); i++) { // if you want to pad: (N + dim - 1) / dim
        /* (row*N): which row we belong to, since one thread travels the whole row -> invariant
        * (i*dim): which step of the tile moving we are at
        * tx: within this step which column am I going to access
        */
        A[ty * dim + tx] = a[(row * N) + (i * dim) + tx];
        /*
        * (i* dim * N): where are we based upon the current iteration of this loop (where is my tile in the y dimension)
        * (ty*N): where are we within the thread block (what row is each thread on)
        * col: column stays the same
        */
        B[ty * dim + tx] = b[(i * dim * N) + (ty * N) + col];
        __syncthreads();

        for (int j = 0; j < dim; j++) {
            temp += A[(ty * dim) + j] * B[(j * dim) + tx];
        }
        __syncthreads(); // Make sure all threads are done before loop starts again (with other warp)
    }

    c[row * N + col] = temp;
    // still need to access global memory but only once per thread block
}

void initMatrix(int *m, int N) {
    for (int i = 0; i < N * N; i++) {
        m[i] = rand() % 100;
    }
}

void verify_result(vector<int> &a, vector<int> &b, vector<int> &c) {
  // For every row...
  for (int i = 0; i < N; i++) {
    // For every column...
    for (int j = 0; j < N; j++) {
      // For every element in the row-column pair
      int tmp = 0;
      for (int k = 0; k < N; k++) {
        // Accumulate the partial results
        tmp += a[i * N + k] * b[k * N + j];
      }

      // Check against the CPU result
      assert(tmp == c[i * N + j]);
    }
  }
}

int main() {
    // Size (in bytes) of matrix
  size_t bytes = N * N * sizeof(int);

  // Host vectors
  vector<int> h_a(N * N);
  vector<int> h_b(N * N);
  vector<int> h_c(N * N);

  // Initialize matrices
  generate(h_a.begin(), h_a.end(), []() { return rand() % 100; });
  generate(h_b.begin(), h_b.end(), []() { return rand() % 100; });

  // Allocate device memory
  int *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, bytes);
  cudaMalloc(&d_b, bytes);
  cudaMalloc(&d_c, bytes);

  // Copy data to the device
  cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice);

  // Threads per CTA dimension
  int THREADS = 32;

  // Blocks per grid dimension (assumes THREADS divides N evenly)
  int BLOCKS = N / THREADS;

  // Use dim3 structs for block  and grid dimensions
  dim3 threads(THREADS, THREADS);
  dim3 blocks(BLOCKS, BLOCKS);

  // Launch kernel
  matrixMul<<<blocks, threads>>>(d_a, d_b, d_c);

  // Copy back to the host
  cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

  // Check result
  verify_result(h_a, h_b, h_c);

  cout << "COMPLETED SUCCESSFULLY\n";

  // Free memory on device
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return 0;
}