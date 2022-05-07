#include "common/book.h"

#define N (30*1024*1024)
#define imin(a,b) (a>b)?b:a;

const int threadsPerBlock = 256;
const int blocksPerBlock = imin(32, (N + threadsPerBlock - 1) / threadsPerBlock);

struct DataBlock
{
	int deviceId;
	int size;
	int offset;
	float* a;
	float* b;
	float returnValue;
};

__global__ void dot(int size, float* a, float* b, float* c)
{
	__shared__ float cache[threadPerBlock];
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int stripe = blockDim.x * gridDim.x;
	int cacheIdx = threadIdx.x;

	float temp = 0;
	while (tid < size)
	{
		temp += a[tid] * b[tid];
		tid += stripe;
	}

	cache[cacheIdx] = temp;
	__syncthreads();

	int i = blockDim.x / 2;
	while (i != 0)
	{
		if (cacheIdx < i)
			cache[cacheIdx] += cache[cacheIdx + i];
		__syncthreads();
	}

	if (cacheIdx == 0) c[blockIdx.x] = cache[0];
}

void* runtime_cu(void * ptr)
{
	DataBlock* data = (DataBlock*)ptr;
	if (data->deviceId != 0)
	{
		cudaSetDevice(data->deviceId);
		cudaSetDeviceFlags(cudaDevcieMapHost);
	}

	int size = data->size;
	float* a, * b, * c;
	float* dev_a, * dev_b, * dev_c;
	a = data->a;
	b = data->b;
	c = (float*)malloc(blocksPerGrid * sizeof(float));

	cudaHostGetDevicePointer(&dev_a, a, 0);
	cudaHostGetDevicePointer(&dev_b, b, 0);
	cudaMalloc((void**)&dev_c, blocksPerGrid * sizeof(float));

	dot << <blocksPerGrim, threadsPerBlock >> > (size, dev_a, dev_b, dev_c);
	cudaMemcpy(c, dev_c, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);


}

int main()
{
	return 0;
}