#include <stdio.h>
#include <math.h>
#include <time.h>
#define M 8
#define N M*M*M
#define NUM 300
typedef float ElemType;

//cpu稠密矩阵乘法
void gemm_cpu(ElemType* A, ElemType* B, ElemType* C, int m, int n)
{
	for (int k = 0; k < n; k++)
	{
		for (int i = 0; i < m; i++)
		{
			int temp = A[i * m + k];
			for (int j = 0; j < n; j++)
			{
				C[i * m + j] += temp * B[k * n + j];
				//for (int i = 0; i < NUM; i++) float tag = 999999 * 1388*i;
			}
		}
	}
}


//每个线程处理矩阵C中一个元素的计算工作--常规方法
__global__ void gemm_1(ElemType* A, ElemType* B, ElemType* C, int m, int n)
{
	int xx = threadIdx.x + blockIdx.x * blockDim.x;
	int yy = threadIdx.y + blockIdx.y * blockDim.y;
	int zz = threadIdx.z + blockIdx.z * blockDim.z;
	//线程id
	int tid = xx + yy * blockDim.x * gridDim.x + zz * blockDim.y * gridDim.y * blockDim.x * gridDim.x;
	
	int bx = tid / n;
	int by = tid % n;
	if (tid < n * m)
	{
		ElemType sum = 0.0;
		for (int i = 0; i < n; i++)
		{
			sum += A[bx * n + i] * B[by * m + i];
			//for (int i = 0; i < NUM; i++) float tag = 999999 * 138882;
		}

		C[bx * n + by] = sum;
	}
}

//每个线程处理矩阵C中一个元素的计算工作--常规方法
__global__ void gemm_tex_1(ElemType* A, ElemType* B, ElemType* C, int m, int n)
{
	int xx = threadIdx.x + blockIdx.x * blockDim.x;
	int yy = threadIdx.y + blockIdx.y * blockDim.y;
	int zz = threadIdx.z + blockIdx.z * blockDim.z;

	int tid = xx + yy * blockDim.x * gridDim.x + zz * blockDim.y * gridDim.y * blockDim.x * gridDim.x;
	int bx = tid / n;
	int by = tid % n;

	if (tid < n * m)
	{
		ElemType sum = 0.0;
		for (int i = 0; i < n; i++)
		{
			sum += A[bx * n + i] * B[by * n + i];
			//for (int i = 0; i < NUM; i++) float tag = 999999 * 138882;
		}

		C[bx * n + by] = sum;
	}
}

//多个线程处理矩阵C中一个元素的计算工作
__global__ void gemm_2(ElemType* A, ElemType* B, ElemType* C, int m, int n, int threads_per_elem)
{
	int xx = threadIdx.x + blockIdx.x * blockDim.x;
	int yy = threadIdx.y + blockIdx.y * blockDim.y;
	int zz = threadIdx.z + blockIdx.z * blockDim.z;

	int tid = xx + yy * blockDim.x * gridDim.x + zz * blockDim.y * gridDim.y * blockDim.x * gridDim.x;

	int wid = tid / threads_per_elem;
	int lane = tid % threads_per_elem;

	int bx = wid / n;
	int by = wid % n;

	if (bx < m)
	{
		ElemType sum = 0.0;
		for (int i = lane; i < n; i += threads_per_elem)
		{
			sum += A[bx*n+i] * B[i*m+by];
			//for (int i = 0; i < NUM; i++) float tag = 999999 * 138882;
		}
		//归约
		int i = threads_per_elem / 2;
		while (i != 0)
		{
			sum += __shfl_down(sum, i);
			i /= 2;
		}

		if (lane == 0) C[bx*n+by] = sum;
	}
}


//一个线程束负责矩阵C中一个元素的计算工作
__global__ void gemm_3(ElemType* A, ElemType* B, ElemType* C, int m, int n, int threads_per_elem)
{
	int xx = threadIdx.x + blockIdx.x * blockDim.x;
	int yy = threadIdx.y + blockIdx.y * blockDim.y;
	int zz = threadIdx.z + blockIdx.z * blockDim.z;

	int tid = xx + yy * blockDim.x * gridDim.x + zz * blockDim.y * gridDim.y * blockDim.x * gridDim.x;

	int wid = tid / threads_per_elem;
	int lane = tid % threads_per_elem;

	int bx = wid / n;
	int by = wid % n;

	if (bx < m)
	{
		ElemType sum = 0.0;
		for (int i = lane; i < n; i += threads_per_elem)
		{
			sum += A[bx * n + i] * B[i * m + by];
			//for (int i = 0; i < NUM; i++) float tag = 999999 * 138882;
		}

		//归约
		sum += __shfl_xor(sum, 16);
		sum += __shfl_xor(sum, 8);
		sum += __shfl_xor(sum, 4);
		sum += __shfl_xor(sum, 2);
		sum += __shfl_xor(sum, 1);

		if (lane == 0)
		{
			C[bx * n + by] = sum;
		}
	}
}

/*
__device__ const int count = 32;

__device__ ElemType warpReduceSum(ElemType sum)
{
	sum += __shfl_down(sum, 16);
	sum += __shfl_down(sum, 8);
	sum += __shfl_down(sum, 4);
	sum += __shfl_down(sum, 2);
	sum += __shfl_down(sum, 1);

	return sum;
}

__device__ ElemType blockReduceSum(ElemType sum, int tid)
{
	__shared__ ElemType temp[count];
	temp[tid] = sum;
	__syncthreads();

	int i = count / 32;
	int wid = tid / 32;
	int lane = tid % 32;
	while (i != 0)
	{
		sum = warpReduceSum(sum);
		if (lane == 0) temp[wid] = sum;
		__syncthreads();

		sum = (threadIdx.x < i) ? temp[threadIdx.x] : 0;
		i /= 32;
		wid /= 32;
		lane = wid % 32;
	}

	return sum;
}

//每个线程都处理一个乘法，如A(N x M) 与 B(M x N) 相乘，用(N x M x N)个线程去处理
__global__ void gemm_4(ElemType* A, ElemType* B, ElemType* C, int n, int m, int t)
{
	int bx = threadIdx.x + blockIdx.x * blockDim.x;
	int by = threadIdx.y + blockIdx.y * blockDim.y;
	int cacheIdx = threadIdx.x + threadIdx.y * blockDim.x;

	if (bx < m && by < n)
	{
		ElemType sum = 0.0;
		for (int i = cacheIdx; i < m; i += count)
		{
			sum += A[by * n + i] * B[i * n + bx];
		}

		sum = blockReduceSum(sum, cacheIdx);

		if (cacheIdx == 0) C[by*n+bx] = sum;
	}
}
*/

//打印矩阵
void printfMatrix(ElemType* A, int m, int n)
{
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			printf("%3.1f ", A[i * m + j]);
		}
		printf("\n");
	}
}

int main()
{
	float elapsedTime;
	//创建事务
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	ElemType* A, * B, * C;
	ElemType* dev_A, * dev_B, * dev_C;
	//主机端内存分配
	int mysize = N * N * sizeof(ElemType);
	A = (ElemType*)malloc(mysize);
	B = (ElemType*)malloc(mysize);
	C = (ElemType*)malloc(mysize);

	//设备端内存分配
	cudaMalloc((void**)&dev_A, mysize);
	cudaMalloc((void**)&dev_B, mysize);
	cudaMalloc((void**)&dev_C, mysize);

	//创建测试用的稠密矩阵
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			A[i * N + j] = 1.0;
			B[i * N + j] = 1.0;
			C[i * N + j] = 0.0;
		}
	}

	//cpu版稠密矩阵乘法
	time_t start_c, stop_c;
	start_c = (unsigned)time(NULL);
	gemm_cpu(A, B, C, N, N);
	stop_c = (unsigned)time(NULL);
	//printfMatrix(C, N, N);
	printf("gemm_cpu spend time is %d s.\n", stop_c-start_c);

	cudaMemcpy(dev_A, A, mysize, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_B, B, mysize, cudaMemcpyHostToDevice);


	//gemm_1函数的计算和优化
	cudaEventRecord(start);
	dim3 gridsize1(M*N/1024, M, M);
	dim3 blocksize1(1024, 1, 1);
	
	gemm_1 << <gridsize1, blocksize1 >> > (dev_A, dev_B, dev_C, N, N);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cudaMemcpy(C, dev_C, mysize, cudaMemcpyDeviceToHost);
	//printfMatrix(C, N, N);
	printf("gemm_1 spend time is %3.1f ms.\n", elapsedTime);


	//gemm_2函数的计算和优化
	cudaEventRecord(start);
	dim3 gridsize2(M * N * 8 / 1024, M, M);
	dim3 blocksize2(1024, 1, 1);

	gemm_2 << <gridsize2, blocksize2 >> > (dev_A, dev_B, dev_C, N, N, 8);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cudaMemcpy(C, dev_C, mysize, cudaMemcpyDeviceToHost);
	//printfMatrix(C, N, N);
	printf("gemm_2 spend time is %3.1f ms.\n", elapsedTime);


	//gemm_3函数的计算和优化
	cudaEventRecord(start);
	dim3 gridsize3(M * N * 32 / 1024, M, M);
	dim3 blocksize3(1024, 1, 1);

	gemm_3 << <gridsize3, blocksize3 >> > (dev_A, dev_B, dev_C, N, N, 32);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cudaMemcpy(C, dev_C, mysize, cudaMemcpyDeviceToHost);
	//printfMatrix(C, N, N);
	printf("gemm_3 spend time is %3.1f ms.\n", elapsedTime);
	return 0;
}