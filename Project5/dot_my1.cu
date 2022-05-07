#include <stdlib.h>
#include <stdio.h>
#include "common/book.h"

#define imin(a,b) (a>b)?b:a;
const int N = 33 * 1024;
const int threadsPerBlock = 256;
const int blocksPerGrid = imin(32, (N + threadsPerBlock - 1) / threadsPerBlock);


__global__ void kernel(float* a, float* b, float* c)
{
	__shared__ float cache[threadsPerBlock];
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int cacheIdx = threadIdx.x;

	float temp = 0;
	while (tid < N)
	{
		temp += a[tid] * b[tid];
		tid += blockDim.x * gridDim.x;
	}

	cache[cacheIdx] = temp;
	__syncthreads();

	int i = blockDim.x/2;
	while (i != 0)
	{
		if (cacheIdx < i) cache[cacheIdx] += cache[cacheIdx + i];
		__syncthreads();
		i /= 2;
	}

	if (cacheIdx == 0) c[blockIdx.x] = cache[0];
}
int main()
{
	float* a, * b, * c;
	float* dev_a, * dev_b, * dev_c;

	a = (float*)malloc(sizeof(float) * N);
	b = (float*)malloc(sizeof(float) * N);
	c = (float*)malloc(sizeof(float) * blocksPerGrid);

	cudaMalloc((void**)&dev_a, sizeof(float) * N);
	cudaMalloc((void**)&dev_b, sizeof(float) * N);
	cudaMalloc((void**)&dev_c, sizeof(float) * blocksPerGrid);

	for (int i = 0; i < N; i++)
	{
		a[i] = i; 
		b[i] = i * 2;
	}

	cudaMemcpy(dev_a, a, sizeof(float) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, sizeof(float) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_c, c, sizeof(float) * blocksPerGrid, cudaMemcpyHostToDevice);

	kernel << <blocksPerGrid, threadsPerBlock >> > (dev_a, dev_b, dev_c);

	cudaMemcpy(c, dev_c, sizeof(float) * blocksPerGrid, cudaMemcpyDeviceToHost);
	float slt = 0;
	for (int i = 0; i < blocksPerGrid; i++)
	{
		slt += c[i];
	}
#define sum_squares(x) (x*(x+1)*(2*x+1))
	printf("GPU does %.6g = %.6g?\n", slt, 2 * sum_squares((float)(N - 1)));

	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_b);
	free(a);
	free(b);
	free(c);
	return 0;
}