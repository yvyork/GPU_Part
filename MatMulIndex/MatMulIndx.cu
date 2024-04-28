#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define SHMEM_SIZE 16*16

__global__ void print_indices(int n) {
    int tile_size = 2; // Assuming the tile size is fixed to 2x2
    int tx = threadIdx.x; 
    int ty = threadIdx.y;
    int bx = blockIdx.x; 
    int by = blockIdx.y;

    // Calculate the row and column in the global matrix for each thread
    int row = by * tile_size + ty; 
    int col = bx * tile_size + tx;

    printf("Block: (%d,%d), Thread: (%d,%d)\n", bx, by, tx, ty);
    printf("tx: %d, ty: %d, bx: %d, by: %d\n", tx, ty, bx, by);
    printf("row: %d, col: %d\n", row, col);

    for (int i = 0; i < (n / tile_size); i++) {
        // Calculate index for A
        int global_row = row * n; // The starting index of the row
        int column_set = i * tile_size + tx;
        int mem_index_a = global_row + column_set;

        // Calculate index for B
        int row_set = i * tile_size; // The starting row for the current tile of B
        int global_col = col; // Column remains the same across different tile sets for B
        int mem_index_b = (row_set + ty) * n + global_col;

        printf("Iteration: %d\n", i);
        printf("Index calculation for A (row start: %d, column set: %d, index: %d)\n", global_row, column_set, mem_index_a);
        printf("Index calculation for B (row set: %d, row within set: %d, index: %d)\n", row_set, ty, mem_index_b);
    }
}


__global__
void cacheTiledMatMul(int* a, int* b, int* c, int width, int tile_size) {


}

int main() {
    int width = 4;  // Assume width is a multiple of 2 for simplicity
    dim3 blocks(width / 2, width / 2); // Configure blocks
    dim3 threads(2, 2); // Configure threads per block

    printf("Matrix Dimension: %dx%d\n", width, width);
    printf("Block Dimension: %dx%d\n", blocks.x, blocks.y);
    printf("Threads per Block: %d\n", threads.x*threads.y);

    // Launch the kernel
    print_indices<<<blocks, threads>>>(width);
    cudaDeviceSynchronize();

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        return -1;
    }

    return 0;
}
