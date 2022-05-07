#include "common/book.h"

#define SIZE (20 * 1024 * 1024)

__global__ void kernel(int* histo, unsigned char* buffer)
{
	__shared__ int temp[256];
	temp[threadIdx.x] = 0;
	__syncthreads();

	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int stripe = blockDim.x * gridDim.x;
	while (tid < SIZE)
	{
		atomicAdd(&(temp[buffer[tid]]), 1);
		tid += stripe;
	}

	__syncthreads();
	atomicAdd(&(histo[threadIdx.x]), temp[threadIdx.x]);
}

int main()
{
	unsigned char* buffer = (unsigned char*)big_random_block(SIZE);

	cudaDeviceProp prop;
	int whichDevice;
	cudaGetDevice(&whichDevice);
	cudaGetDeviceProperties(&prop, whichDevice);
	int blocks = 2 * prop.multiProcessorCount;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	int* histo, * dev_histo;
	unsigned char* dev_buffer;
	
	histo = (int*)malloc(256 * sizeof(int));
	cudaMalloc((void**)&dev_histo, 256 * sizeof(int));
	cudaMalloc((void**)&dev_buffer, SIZE * sizeof(int));
	cudaMemcpy(dev_buffer, buffer, SIZE * sizeof(unsigned char), cudaMemcpyHostToDevice);

	kernel << <blocks, 256 >> > (dev_histo, dev_buffer);

	cudaMemcpy(histo, dev_histo, 256 * sizeof(int), cudaMemcpyDeviceToHost);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("average time is : %3.1f ms.\n", elapsedTime);

	for (int i = 0; i < SIZE; i++) histo[buffer[i]]--;
	for (int i = 0; i < 256; i++) if (histo[i] != 0) printf("Failure\n");

	cudaFree(dev_buffer);
	cudaFree(dev_histo);
	free(histo);
	free(buffer);
	return 0;
}