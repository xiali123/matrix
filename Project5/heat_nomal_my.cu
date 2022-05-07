#include "cuda.h"
#include "common/book.h"
#include "common/cpu_anim.h"

#define DIM 1024
#define MAX_TEMP 1.0f
#define MIN_TEMP 0.00001f
#define SPEED 0.25f

struct DataBlock
{
	unsigned char* output_bitmap;
	float* dev_inStr;
	float* dev_outStr;
	float* dev_constStr;
	CPUAnimBitmap* bitmap;
	cudaEvent_t start, stop;
	float totalTime;
	float frames;
};

__global__ void blend_kernel(float* dstOut, const float* dstIn)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;
	
	int left = offset - 1;
	int right = offset + 1;
	if (x == 0) left++;
	if (x == DIM - 1) right--;

	int top = offset - DIM;
	int bottom = offset + DIM;
	if (y == 0) top += DIM;
	if (y == DIM - 1) bottom -= DIM;

	dstOut[offset] = dstIn[offset] + SPEED * (dstIn[left] + dstIn[right] + dstIn[top] + dstIn[bottom] - 4 * dstIn[offset]);
}

__global__ void copy_const_kernel(float* iptr, const float* cptr)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	float c = cptr[offset];
	if (c != 0) iptr[offset] = c;
}

void anim_gpu(DataBlock* d, int intr)
{
	cudaEventRecord(d->start, 0);
	dim3 blocks(DIM / 16, DIM / 16);
	dim3 threads(16, 16);
	CPUAnimBitmap* bitmap = d->bitmap;

	for (int i = 0; i < 90; i++)
	{
		copy_const_kernel << <blocks, threads >> > (d->dev_inStr, d->dev_constStr);
		blend_kernel << <blocks, threads >> > (d->dev_outStr, d->dev_inStr);
		swap(d->dev_inStr, d->dev_outStr);
	}
	float_to_color << <blocks, threads >> > (d->output_bitmap, d->dev_inStr);
	cudaMemcpy(bitmap->get_ptr(), d->output_bitmap, bitmap->image_size(), cudaMemcpyDeviceToHost);

	cudaEventRecord(d->stop, 0);
	cudaEventSynchronize(d->stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, d->start, d->stop);
	d->totalTime += elapsedTime;
	++d->frames;

	printf("Average time is : %3.1f ms \n", d->totalTime / d->totalTime);
}

void anim_exit(DataBlock* d)
{
	cudaFree(d->dev_inStr);
	cudaFree(d->dev_outStr);
	cudaFree(d->dev_constStr);

	cudaEventDestroy(d->start);
	cudaEventDestroy(d->stop);
}

int main(void)
{
	DataBlock data;
	CPUAnimBitmap bitmap(DIM, DIM, &data);
	data.bitmap = &bitmap;
	data.totalTime = 0; 
	data.frames = 0;

	cudaEventCreate(&data.start);
	cudaEventCreate(&data.stop);

	int imageSize = bitmap.image_size();

	cudaMalloc((void**)&data.output_bitmap, imageSize);
	cudaMalloc((void**)&data.dev_inStr, imageSize);
	cudaMalloc((void**)&data.dev_outStr, imageSize);
	cudaMalloc((void**)&data.dev_constStr, imageSize);

	float* temp = (float*)malloc(imageSize);
	for (int i = 0; i < DIM * DIM; i++)
	{
		temp[i] = 0; 
		int x = i % DIM;
		int y = i / DIM;
		if (x > 300 && x < 600 && y > 300 && y < 600) temp[i] = MAX_TEMP;
	}

	temp[DIM * 100 + 100] = (MIN_TEMP + MAX_TEMP) / 2;
	temp[DIM * 300 + 300] = MIN_TEMP;
	temp[DIM * 700 + 200] = MIN_TEMP;
	temp[DIM * 300 + 600] = MIN_TEMP;

	for (int y = 800; y < 900; y++)
	{
		for (int x = 400; x < 500; x++) temp[x + y * DIM] = MIN_TEMP;
	}

	cudaMemcpy(data.dev_constStr, temp, imageSize, cudaMemcpyHostToDevice);

	for (int y = 800; y < DIM; y++)
	{
		for (int x = 0; x < 200; x++) temp[x + y * DIM] = MAX_TEMP;
	}
	cudaMemcpy(data.dev_inStr, temp, imageSize, cudaMemcpyHostToDevice);
	free(temp);

	bitmap.anim_and_exit((void(*)(void*, int))anim_gpu, (void(*)(void*))anim_exit);
}