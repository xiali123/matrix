#include "common/book.h"
#include "common/gpu_anim.h"

#define DIM 1024

__global__ void kernel(uchar4* ptr, int intr)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	int fx = (x - DIM / 2);
	int fy = (y - DIM / 2);
	float d = sqrtf(fx * fx + fy * fy);
	float gray = 128.0f + 127.0f * cos(d / 10.0f - intr / 7.0f);
	ptr[offset].x = gray;
	ptr[offset].y = gray;
	ptr[offset].z = gray;
	ptr[offset].w = 255;
}

void generate_frame(uchar4* ptr, void *, int intr)
{
	dim3 blocks(DIM / 16, DIM / 16);
	dim3 threads(16, 16);
	kernel << <blocks, threads >> > (ptr, intr);
}

int main()
{
	GPUAnimBitmap bitmap(DIM, DIM, NULL);
	bitmap.anim_and_exit((void(*)(uchar4*, void*, int))generate_frame, NULL);
	return 0;
}