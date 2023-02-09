#include <stdio.h>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 256

struct matrix {
        unsigned long int height;
        unsigned long int width;
        float *h_rows;
        float *d_rows;
};

__global__
void kernel_scalar_matrix_mult(float scalar_value, struct matrix matrix, unsigned long int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (unsigned long int i = index; i < n; i += stride) {
	matrix.d_rows[i] = matrix.d_rows[i] * scalar_value;
    }
}

__global__
void kernel_matrix_matrix_mult(struct matrix a, struct matrix b, struct matrix c) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int pos;

    unsigned long int i, j, k, n;

    n = c.height * c.width;

    for (pos = index; pos <  n; pos += stride) {
    	i = pos / c.width;
    	j = pos % c.width;

	c.d_rows[pos] = 0;

	/* Proccess the product between each element of the row of matrix a  */
	/* and each element of the colum of matrix b and accumulates the sum */
	/* of the product on the correspondent element of matrix c.          */
	for (k = 0; k < a.width; ++k) {
		c.d_rows[pos] += a.d_rows[(i * a.width) + k] * b.d_rows[(k * b.height) + j];
	}
    }
}

int scalar_matrix_mult(float scalar_value, struct matrix *matrix) {
  unsigned long int N;
  float *d_rows;
  int blockSize, numBlocks;
  cudaError_t cudaError;

  /* Check the numbers of the elements of the matrix */
  N = matrix->height * matrix->width;

  /* Check the integrity of the matrix */
  if (N == 0 || matrix->h_rows == NULL) return 0;

  /* Allocate memory on device to process the product */
  cudaError = cudaMalloc(&d_rows, N*sizeof(float));

  // check cudaMalloc memory allocation
  if (cudaError != cudaSuccess) {
	printf("cudaMalloc d_rows returned error %s (code %d)\n", cudaGetErrorString(cudaError), cudaError);
        return 0;
  }

  /* Copy the rows of the matrix to device */
  cudaError = cudaMemcpy(d_rows, matrix->h_rows, N*sizeof(float), cudaMemcpyHostToDevice);

  if (cudaError != cudaSuccess) {
	printf("cudaMemcpy (h_rows -> d_rows) returned error %s (code %d), line(%d)\n", cudaGetErrorString(cudaError), cudaError, __LINE__);
        return 0;
  }

  // Setting temp matrix
  matrix->d_rows = d_rows;

  // Run kernel on elements on the GPU
  blockSize = THREADS_PER_BLOCK;
  numBlocks = (N + blockSize - 1) / blockSize;
  kernel_scalar_matrix_mult<<<numBlocks, blockSize>>>(scalar_value, *(matrix), N);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  /* Copy rows from device back to matrix */
  cudaError = cudaMemcpy(matrix->h_rows, d_rows, N*sizeof(float), cudaMemcpyDeviceToHost);

  if (cudaError != cudaSuccess)
  {
	printf("cudaMemcpy (d_rows -> h_rows) returned error %s (code %d), line(%d)\n", cudaGetErrorString(cudaError), cudaError, __LINE__);
	return 1;
  }

  /* Freeing device memory */
  cudaFree(d_rows);

  return 1;
}

int matrix_matrix_mult(struct matrix *a, struct matrix *b, struct matrix *c) {
  unsigned long int NA, NB, NC;
  int blockSize, numBlocks;
  cudaError_t cudaError;

  /* Check the numbers of the elements of the matrix */
  NA = a->height * a->width;
  NB = b->height * b->width;
  NC = c->height * c->width;

  /* Check the integrity of the matrix */
  if ( (NA == 0 || a->h_rows == NULL) ||
       (NB == 0 || b->h_rows == NULL) ||
       (NC == 0 || c->h_rows == NULL) ) return 0;

  /* Check if we can execute de product of matrix A and matrib B */
  if ( (a->width != b->height) ||
       (c->height != a->height) ||
       (c->width != b->width) ) return 0;

  /* Allocate memory on device to process the product */
  cudaError = cudaMalloc(&a->d_rows, NA*sizeof(float));

  // check cudaMalloc memory allocation
  if (cudaError != cudaSuccess) {
	printf("cudaMalloc a->d_rows returned error %s (code %d)\n", cudaGetErrorString(cudaError), cudaError);
        return 0;
  }

  /* Allocate memory on device to process the product */
  cudaError = cudaMalloc(&b->d_rows, NB*sizeof(float));

  // check cudaMalloc memory allocation
  if (cudaError != cudaSuccess) {
	printf("cudaMalloc b->d_rows returned error %s (code %d)\n", cudaGetErrorString(cudaError), cudaError);
        return 0;
  }

  /* Allocate memory on device to process the product */
  cudaError = cudaMalloc(&c->d_rows, NC*sizeof(float));

  // check cudaMalloc memory allocation
  if (cudaError != cudaSuccess) {
	printf("cudaMalloc c->d_rows returned error %s (code %d)\n", cudaGetErrorString(cudaError), cudaError);
        return 0;
  }

  /* Copy the rows of the matrix to device */
  cudaError = cudaMemcpy(a->d_rows, a->h_rows, NA*sizeof(float), cudaMemcpyHostToDevice);

  if (cudaError != cudaSuccess) {
	printf("cudaMemcpy (a.h_rows -> a.d_rows) returned error %s (code %d), line(%d)\n", cudaGetErrorString(cudaError), cudaError, __LINE__);
        return 0;
  }

  /* Copy the rows of the matrix to device */
  cudaError = cudaMemcpy(b->d_rows, b->h_rows, NB*sizeof(float), cudaMemcpyHostToDevice);

  if (cudaError != cudaSuccess) {
	printf("cudaMemcpy (b.h_rows -> b.d_rows) returned error %s (code %d), line(%d)\n", cudaGetErrorString(cudaError), cudaError, __LINE__);
        return 0;
  }

  // Run kernel on elements on the GPU
  blockSize = THREADS_PER_BLOCK;
  numBlocks = (NC + blockSize - 1) / blockSize;
  kernel_matrix_matrix_mult<<<numBlocks, blockSize>>>(*a, *b, *c);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  /* Copy rows from device back to matrix */
  cudaError = cudaMemcpy(c->h_rows, c->d_rows, NC*sizeof(float), cudaMemcpyDeviceToHost);

  if (cudaError != cudaSuccess)
  {
	printf("cudaMemcpy (c.d_rows -> c.h_rows) returned error %s (code %d), line(%d)\n", cudaGetErrorString(cudaError), cudaError, __LINE__);
	return 1;
  }

  /* Freeing device memory */
  cudaFree(a->d_rows);
  cudaFree(b->d_rows);
  cudaFree(c->d_rows);

  return 1;
}
