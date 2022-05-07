#include "common/book.h"

#define N (30*1024*1024)
#define imin(a,b) (a>b)?b:a;
const int threadsPerBlock = 256;
const int blocksPerGrid = imin(32,(N+threadsPerBlock-1)/threadsPerBlock);

struct DataBlock
{
	int deviceId;
	float* a;
	float* b;
	
};


__global__ void dot(int size, float* a, float* b, float* c)
{
	__shared__ float cache[threadsPerBlock];
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int stripe = blockDim.x * gridDim.x;
	int cacheIdx = threadsIdx.x;
	float temp = 0;

	while (tid < size)
	{
		temp += a[tid] * b[tid];
		tid += stripe;
	}

	cache[cacheIdx] = temp;
	__syncthreads();

	int i = blockDim.x;
	while (i != 0)
	{
		if (cacheIdx < i)
			cache[cacheIdx] += cache[cacheIdx + i];
		__syncthreads();
	}

	if (cacheIdx == 0) c[blockIdx.x] = cache[0];
}

void* runtime(void* ptr)
{
	DataBlock* data = (DataBlock*)ptr;
	
	if (data.deviceIdx != 0)
	{
		cudaSetDevice(data.deviceIdx);
		cudaSetDeviceFlags(cudaDeviceMapHost);
	}

	int size = data->size;
	float* a, * b, * c;
	float* dev_a, * dev_b, * dev_c;

	a = data->a;
	b = data->b;
	c = (float*)malloc(sizeof(float) * blocksPerGrid);

	cudaHostGetDevicePointer(&dev_a, a, 0);
	cudaHostGetDevicePointer(&dev_b, b, 0);
	cudaMalloc((void**)&dev_c, blocksPerGrid * sizeof(float));

	dev_a += data->offset;
	dev_b += data->offset;

	dot << < blocksPerGrid, threadsPerBlock >> > (size, dev_a, dev_b, dev_c);
	cudeMemcpy(c, dev_c, blocksPerGrid*sizeof(float), cudaMemcpyDevcieToHost);

	float cc = 0;
	for (int i = 0; i < blocksPerGrid; i++)
	{
		cc += c[i];
	}
	cudaFree(dev_c);
	free(c);
	return 0;
}

int main()
{

	return 0;
}