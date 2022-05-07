#include "cuda.h"
#include "common/book.h"
#include "common/cpu_anim.h"

#define DIM 1024
#define MIN_TEMP 0.00001f
#define MAX_TEMP 1.0f
#define SPEED 0.25f

texture<float, 2> texIn;
texture<float, 2> texOut;
texture<float, 2> texConstStr;

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

__global__ void copy_const_kernel(float* iptr)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	float c = tex2D(texConstStr, x, y);
	if (c != 0) iptr[offset] = c;
}

__global__ void blend_kernel(float* dst, bool dstOut)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	float l, r, t, b, c;
	if (dstOut)
	{
		l = tex2D(texIn, x - 1, y);
		r = tex2D(texIn, x + 1, y);
		t = tex2D(texIn, x, y - 1);
		b = tex2D(texIn, x, y + 1);
		c = tex2D(texIn, x, y);
	}
	else
	{
		l = tex2D(texOut, x - 1, y);
		r = tex2D(texOut, x + 1, y);
		t = tex2D(texOut, x, y - 1);
		b = tex2D(texOut, x, y + 1);
		c = tex2D(texOut, x, y);
	}
	dst[offset] = c + SPEED * (l + r + t + b - 4 * c);
}

void anim_gpu(DataBlock* d, int intr)
{
	cudaEventRecord(d->start, 0);
	dim3 blocks(DIM / 16, DIM / 16);
	dim3 threads(16, 16);
	CPUAnimBitmap* bitmap = d->bitmap;
	volatile bool dstOut = true;
	for (int i = 0; i < 90; i++)
	{
		float* in, * out;
		if (dstOut)
		{
			in = d->dev_inStr;
			out = d->dev_outStr;
		}
		else
		{
			in = d->dev_outStr;
			out = d->dev_inStr;
		}
		copy_const_kernel << <blocks, threads >> > (in);
		blend_kernel << <blocks, threads >> > (out, dstOut);
		dstOut = !dstOut;
	}

	float_to_color << <blocks, threads >> > (d->output_bitmap, d->dev_inStr);
	cudaMemcpy(bitmap->get_ptr(), d->output_bitmap, bitmap->image_size(), cudaMemcpyDeviceToHost);

	cudaEventRecord(d->stop, 0);
	cudaEventSynchronize(d->stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, d->start, d->stop);
	d->totalTime += elapsedTime;
	++d->frames;
	printf("Average time is : %3.1f ms\n", d->totalTime / d->frames);
}

void anim_exit(DataBlock* d)
{
	cudaUnbindTexture(texIn);
	cudaUnbindTexture(texOut);
	cudaUnbindTexture(texConstStr);

	cudaFree(d->dev_inStr);
	cudaFree(d->dev_outStr);
	cudaFree(d->dev_constStr);

	cudaEventDestroy(d->start);
	cudaEventDestroy(d->stop);
}

int main()
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

	cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
	HANDLE_ERROR(cudaBindTexture2D(NULL, texConstStr,
		data.dev_constStr,
		desc, DIM, DIM,
		sizeof(float) * DIM));

	HANDLE_ERROR(cudaBindTexture2D(NULL, texIn,
		data.dev_inStr,
		desc, DIM, DIM,
		sizeof(float) * DIM));

	HANDLE_ERROR(cudaBindTexture2D(NULL, texOut,
		data.dev_outStr,
		desc, DIM, DIM,
		sizeof(float) * DIM));

	float* temp = (float*)malloc(imageSize);
	for (int i = 0; i < DIM * DIM; i++)
	{
		temp[i] = 0; 
		int x = i % DIM;
		int y = i / DIM;
		if (x > 300 && x < 600 && y > 300 && y < 600) temp[i] = MAX_TEMP;
	}

	temp[DIM * 100 + 100] = (MAX_TEMP + MIN_TEMP) / 2;
	temp[DIM * 700 + 100] = MIN_TEMP;
	temp[DIM * 300 + 300] = MIN_TEMP;
	temp[DIM * 200 + 700] = MIN_TEMP;

	for (int y = 800; y < 900; y++)
	{
		for (int x = 400; x < 500; x++)
		{
			temp[x + y * DIM] = MIN_TEMP;
		}
	}

	cudaMemcpy(data.dev_constStr, temp, imageSize, cudaMemcpyHostToDevice);
	for (int y = 800; y < DIM; y++)
	{
		for (int x = 0; x < 200; x++) temp[x + y * DIM] = MAX_TEMP;
	}
	cudaMemcpy(data.dev_inStr, temp, imageSize, cudaMemcpyHostToDevice);
	free(temp);
	bitmap.anim_and_exit((void(*)(void*, int))anim_gpu, (void(*)(void*))anim_exit);
	return 0;
}