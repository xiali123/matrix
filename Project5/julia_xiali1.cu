#include "common/book.h"
#include "common/cpu_bitmap.h"

#define DIM 1000

struct cuComplex
{
	float r;
	float i;
	__device__ cuComplex(float a, float b): r(a), i(b){}
	__device__ float magnitude2()
	{
		return r * r + i * i;
	}
	__device__ cuComplex operator*(const cuComplex& a)
	{
		return cuComplex(a.r * r - a.i * i, a.r * i + a.i * r);
	}
	__device__ cuComplex operator+(const cuComplex& a)
	{
		return cuComplex(a.r + r, a.i + i);
	}
};

__device__ int julia(int x, int y)
{
	const float scale = 1.5;
	float fx = scale * (float)(DIM / 2 - x) / (DIM / 2);
	float fy = scale * (float)(DIM / 2 - y) / (DIM / 2);
	cuComplex c(-0.8, 0.156);
	cuComplex a(fx, fy);

	for (int i = 0; i < 200; i++)
	{
		a = a * a + c;
		if (a.magnitude2() > 1000) return 0;
	}

	return 1;
}
__global__ void kernel(unsigned char* ptr)
{
	int x = blockIdx.x;
	int y = blockIdx.y;
	int offset = x + y * gridDim.x;

	int myvalue = julia(x, y);
	ptr[offset * 4 + 0] = 255 * myvalue;
	ptr[offset * 4 + 1] = 0;
	ptr[offset * 4 + 2] = 0;
	ptr[offset * 4 + 3] = 255;
}

struct BlockData
{
	unsigned char* dev_bitmap;
};

int main(void)
{
	BlockData data;

	CPUBitmap bitmap(DIM, DIM, &data);
	unsigned char* dev_bitmap;

	cudaMalloc((void**)&dev_bitmap, bitmap.image_size());
	data.dev_bitmap = dev_bitmap;

	dim3 grid(DIM, DIM);
	kernel << <grid, 1 >> > (dev_bitmap);
	cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost);

	cudaFree(dev_bitmap);

	bitmap.display_and_exit();
	return 0;
}