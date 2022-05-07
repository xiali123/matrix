#include "common/book.h"

#define imin(a,b) (a>b)?b:a;
#define     N    (33*1024*1024)
const int threadsPerBlock = 256;
const int blocksPerGrid = imin(32,(N+threadsPerBlock-1)/ threadsPerBlock);

struct DataBlock
{
	int deviceIdx;
	float* a;
	float* b;
	int size;
	int offset;
	float returnValue;
};

__global__ void dot(int size, float *a, float *b, float *c)
{
	__shared__ float cache[threadsPerBlock];
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
			cache[cacheIdx] += chache[chacheIdx + i];
		__syncthreads();
	}

	if (cacheIdx == 0) c[blockIdx.x] = cache[0];
}

void* runtime_cu(void* pvoidData)
{
	DataBlock* data = (DataBlock*)pvoidData;
	if (data->deviceIdx != 0)
	{
		cudaSetDevice(data->deviceIdx);
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
	dot << <threadsPerBlock, blocksPerGrid >> > (size, dev_a, dev_b, dev_c);

	cudaMemcpy(c, dev_c, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);

	float returnC = 0;
	for (int i = 0; i < blocksPerGrid; i++)
	{
		returnC += c[i];
	}

	cudaFree(dev_c);
	free(c);

	data->returnValue = returnC;
	return 0;
}

int main()
{
	int deviceCount;
	cudaGetDeviceCount(&count);
	if (deviceCount < 2)
	{
		printf("this device is just one device.\n");
		return 0;
	}

	cudaDeviceProp prop;
	for (int i = 0; i < deviceCount; i++)
	{
		cudaGetDeviceProperties(&prop, 0);
		if (prop.canMapHostMemory != 1)
		{
			printf("this device is not support MapHostMemoy.\n");
			return 0;
		}
	}
	
	float* a, * b;
	cudaSetDevice(0);
	cudaSetDevcieMapFlags(cudaDeviceMapHost);
	cudaHostAlloc((void**)&a, N * sizeof(float), cudaHostAllocWriteCombined | cudaHostAllocProtable | cudaHostAllocMapped);
	cudaHostAlloc((void**)&b, N * sizeof(float), cudaHostAllocWriteCombined | cudaHostAllocPortable | cudaHostAllocMapped);

	for (int i = 0; i < N; i++)
	{
		a[i] = i;
		b[i] = i * 2;
	}

	DataBlock data[2];
	data[0].deviceIdx = 0;
	data[0].size = N / 2;
	data[0].a = a;
	data[0].b = b;
	data[0].offset = 0;
	
	data[1].deviceIdx = 1;
	data[1].size = N / 2;
	data[1].a = a;
	data[1].b = b;
	data[1].offset = N / 2;
	
	CUTThread thread = start_thread(runtime_cu, &data[1]);
	runtime_cu(&data[0]);
	end_thread(thread);

	cudaFreeHost(a);
	cudaFreeHost(b);

	printf("Value calculated:  %f\n", data[0].returnValue + data[1].returnValue);
	return 0;
}