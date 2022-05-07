#include "cuda.h"
#include "common/book.h"
#include "common/cpu_anim.h"

#define DIM 1024
#define PI 3.1415926535897932f
#define MAX_TEMP 1.0f
#define MIN_TEMP 0.0001f
#define SPEED   0.25f

texture<float> texIn;
texture<float> texOut;
texture<float> texConstStr;

struct DataBlock
{
	unsigned char* output_bitmap;
	CPUAnimBitmap* bitmap;
	float* dev_outStr;
	float* dev_inStr;
	float* dev_constStr;
	cudaEvent_t start, stop;
	float totalTime;
	float frames;
};


__global__ void copy_const_kernel(float* iptr) 
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	float c = tex1Dfetch(texConstStr, offset);
	if (c != 0)
	{
		iptr[offset] = c;
	}
}

__global__ void blend_kernel(float* dst,
	bool dstOut) {
	// map from threadIdx/BlockIdx to pixel position
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	int left = offset - 1;
	int right = offset + 1;
	if (x == 0)   left++;
	if (x == DIM - 1) right--;

	int top = offset - DIM;
	int bottom = offset + DIM;
	if (y == 0)   top += DIM;
	if (y == DIM - 1) bottom -= DIM;

	float   t, l, c, r, b;
	if (dstOut) {
		t = tex1Dfetch(texIn, top);
		l = tex1Dfetch(texIn, left);
		c = tex1Dfetch(texIn, offset);
		r = tex1Dfetch(texIn, right);
		b = tex1Dfetch(texIn, bottom);

	}
	else {
		t = tex1Dfetch(texOut, top);
		l = tex1Dfetch(texOut, left);
		c = tex1Dfetch(texOut, offset);
		r = tex1Dfetch(texOut, right);
		b = tex1Dfetch(texOut, bottom);
	}
	dst[offset] = c + SPEED * (t + b + r + l - 4 * c);
}

void anim_gpu(DataBlock* d, int ticks) {
	HANDLE_ERROR(cudaEventRecord(d->start, 0));
	dim3    blocks(DIM / 16, DIM / 16);
	dim3    threads(16, 16);
	CPUAnimBitmap* bitmap = d->bitmap;

	// since tex is global and bound, we have to use a flag to
	// select which is in/out per iteration
	volatile bool dstOut = true;
	for (int i = 0; i < 90; i++) {
		float* in, * out;
		if (dstOut) {
			in = d->dev_inStr;
			out = d->dev_outStr;
		}
		else {
			out = d->dev_inStr;
			in = d->dev_outStr;
		}
		copy_const_kernel << <blocks, threads >> > (in);
		blend_kernel << <blocks, threads >> > (out, dstOut);
		dstOut = !dstOut;
	}
	float_to_color << <blocks, threads >> > (d->output_bitmap,
		d->dev_inStr);

	HANDLE_ERROR(cudaMemcpy(bitmap->get_ptr(),
		d->output_bitmap,
		bitmap->image_size(),
		cudaMemcpyDeviceToHost));

	HANDLE_ERROR(cudaEventRecord(d->stop, 0));
	HANDLE_ERROR(cudaEventSynchronize(d->stop));
	float   elapsedTime;
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime,
		d->start, d->stop));
	d->totalTime += elapsedTime;
	++d->frames;
	printf("Average Time per frame:  %3.1f ms\n",
		d->totalTime / d->frames);
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
	
	cudaBindTexture(NULL, texConstStr, data.dev_constStr, imageSize);
	cudaBindTexture(NULL, texIn, data.dev_inStr, imageSize);
	cudaBindTexture(NULL, texOut, data.dev_outStr, imageSize);

	float* temp = (float*)malloc(imageSize);
	for (int i = 0; i < DIM * DIM; i++) {
		temp[i] = 0;
		int x = i % DIM;
		int y = i / DIM;
		if ((x > 300) && (x < 600) && (y > 310) && (y < 601))
			temp[i] = MAX_TEMP;
	}
	temp[DIM * 300 + 200] = (MAX_TEMP + MIN_TEMP) / 2;
	temp[DIM * 600 + 700] = MIN_TEMP;
	temp[DIM * 300 + 300] = MIN_TEMP;
	temp[DIM * 600 + 300] = MIN_TEMP;
	for (int y = 800; y < 900; y++) {
		for (int x = 400; x < 500; x++) {
			temp[x + y * DIM] = MIN_TEMP;
		}
	}

	cudaMemcpy(data.dev_constStr, temp, imageSize, cudaMemcpyHostToDevice);

	for (int y = 800; y < DIM; y++) {
		for (int x = 0; x < 200; x++) {
			temp[x + y * DIM] = MAX_TEMP;
		}
	}

	cudaMemcpy(data.dev_inStr, temp, imageSize, cudaMemcpyHostToDevice);
	free(temp);

	bitmap.anim_and_exit((void (*)(void*, int))anim_gpu,
		(void (*)(void*))anim_exit);
}