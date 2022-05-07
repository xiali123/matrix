#include "common/book.h"

#define N (1024*1024)
#define FULL_DATA_SIZE (N*20)

__global__ void kernel(int* dev_a, int* dev_b, int* dev_c)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if(idx < N)
	{
		int idx1 = (idx + 1) % 256;
		int idx2 = (idx + 2) % 256;
		float as = (dev_a[idx] + dev_a[idx1] + dev_a[idx2]) / 3.0f;
		float bs = (dev_b[idx] + dev_b[idx1] + dev_b[idx2]) / 3.0f;
		dev_c[idx] = (as + bs) / 2;
	}
}

int main(void)
{
	cudaDeviceProp prop;
	int whichDevice;

	cudaGetDevice(&whichDevice);
	cudaGetDeviceProperties(&prop, whichDevice);

	if (!prop.deviceOverlap)
	{
		printf("Device will not handle overlaps, so no speed up from streams\n");
		return 0;
	}

	cudaEvent_t start, stop;
	cudaStream_t stream;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaStreamCreate(&stream);
	
	int* dev_a, * dev_b, * dev_c;
	int* host_a, * host_b, * host_c;

	cudaMalloc((void**)&dev_a, N * sizeof(int));
	cudaMalloc((void**)&dev_b, N * sizeof(int));
	cudaMalloc((void**)&dev_c, N * sizeof(int));

	cudaHostAlloc((void**)&host_a, FULL_DATA_SIZE * sizeof(int), cudaHostAllocDefault);
	cudaHostAlloc((void**)&host_b, FULL_DATA_SIZE * sizeof(int), cudaHostAllocDefault);
	cudaHostAlloc((void**)&host_c, FULL_DATA_SIZE * sizeof(int), cudaHostAllocDefault);

	for (int i = 0; i < FULL_DATA_SIZE; i++)
	{
		host_a[i] = rand();
		host_b[i] = rand();
	}

	cudaEventRecord(start, 0);

	for (int i = 0; i < FULL_DATA_SIZE; i += N)
	{
		cudaMemcpyAsync(dev_a, host_a + i, N * sizeof(int), cudaMemcpyHostToDevice, stream);
		cudaMemcpyAsync(dev_b, host_b + i, N * sizeof(int), cudaMemcpyHostToDevice, stream);

		kernel << <N / 256, 256, 0, stream >> > (dev_a, dev_b, dev_c);
		cudaMemcpyAsync(host_c + i, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost, stream);
	}


	cudaStreamSynchronize(stream);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	
	float elapsedTime;

	cudaEventElapsedTime(&elapsedTime, start, stop);

	printf("Time taken:  %3.1f ms\n", elapsedTime);

	cudaFreeHost(host_a);
	cudaFreeHost(host_b);
	cudaFreeHost(host_c);
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaStreamDestroy(stream);
	return 0;
}