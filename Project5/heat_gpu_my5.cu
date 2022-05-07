#include "common/book.h"
#include "common/gpu_anim.h"

#define DIM 1024

__global__ void kernel(uchar4* ptr, int intr)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	float fx = x - DIM / 2;
	float fy = y - DIM / 2;
	float c = sqrtf(fx * fx + fy * fy);

	unsigned char green = 128.0f + 127.0f * cos(c / 3.0f - intr / 7.0f);
	ptr[offset].x = green;
	ptr[offset].y = green;
	ptr[offset].z = green;
	ptr[offset].w = 255;
}

void genarate_frame(uchar4* pixels, void*, int intr)
{
	dim3 blocks(DIM / 16, DIM / 16);
	dim3 threads(16, 16);
	kernel << <blocks, threads >> > (pixels, intr);
}

int main()
{
	GPUAnimBitmap bitmap(DIM, DIM, NULL);
	bitmap.anim_and_exit((void(*)(uchar4*, void*, int))genarate_frame, NULL);
}