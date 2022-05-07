/**
* Author: LiXia
* Date: 2022.04.28
**/
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include "cuda.h"
#include "common/book.h"

#define N 32
#define M 1024*16

#define BLOCK_SIZE 1024*4
#define imin(a,b) (a>b)?b:a;

void spmv_cpu(float* value, int* col, int* row, float* x, float* y)
{
	for (int i = 0; i < M; i++)
	{
		int start = row[i];
		int end = row[i + 1];
		float sum = 0.0;
		for (int j = start; j < end; j++)
		{
			sum += (value[j] * x[col[j]]);
		}

		y[i] = sum;
	}
}

//每个线程处理一行的数据
__global__ void spmv_csr1(float* value, int* col, int* row, float* x, float* y)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid <M)
	{
		int row_start = row[tid];
		int row_end = row[tid + 1];
		float sum = 0.0;
		for (int i = row_start; i < row_end; i++)
		{
			sum += (value[i] * x[col[i]]);
		}
		//printf("thread=%d, block= %d,sum%d = %3.1f\n", threadIdx.x, blockIdx.x, tid, sum);
		y[tid] = sum;
	}
}


//每个线程处理一行的数据(共享内存版本)
__global__ void spmv_csr1_shared(const float* value, int* col, int* row, const float* x, float* y)
{
	__shared__ float tempx[BLOCK_SIZE];
	int block_start = blockIdx.x * blockDim.x;
	int tid = threadIdx.x + block_start;
	
	int cacheid = threadIdx.x;
	while (cacheid * 4 < BLOCK_SIZE && cacheid < M)
	{
		tempx[cacheid] = x[cacheid];
		cacheid += blockDim.x;
	}
	__syncthreads();

	if (tid < M)
	{
		int row_start = row[tid];
		int row_end = row[tid + 1];
		float sum = 0.0;
		for (int i = row_start; i < row_end; i++)
		{
			float tag = (col[i] >= BLOCK_SIZE) ? x[col[i]] : tempx[col[i]];		//如果处在共享内存内，则从共享内存中取数
			sum += (value[i] * tag);
		}

		y[tid] = sum;
	}
}


//一个线程束协同处理一行的数据
__global__ void spmv_csr2(float* value, int* col, int* row, float* x, float* y)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	int warpID = tid / 32;
	int lanedID = tid % 32;

	if (warpID < M)
	{
		int start = row[warpID];
		int end = row[warpID + 1];

		//if (lanedID == 0) printf(" start=%d, end= %d, tid= %d\n", start, end, warpID);
		float sum = 0.0;
		for (int i = start + lanedID; i < end; i += 32)
		{
			sum += (value[i] * x[col[i]]);
		}

		//__syncthreads();
		//归约
		sum += __shfl_down(sum, 16);
		sum += __shfl_down(sum, 8);
		sum += __shfl_down(sum, 4);
		sum += __shfl_down(sum, 2);
		sum += __shfl_down(sum, 1);

		//if(lanedID == 0) printf(" thread=%d, block= %d,sum%d = %3.1f\n",threadIdx.x, blockIdx.x, tid, sum);
		if (lanedID == 0) y[warpID] = sum;
	}
}


//一个线程束协同处理一行的数据(共享内存版本)
__global__ void spmv_csr2_shared(float* value, int* col, int* row, float* x, float* y)
{
	__shared__ float tempx[BLOCK_SIZE];
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	int warpID = tid / 32;
	int lanedID = tid % 32;

	int cacheid = threadIdx.x;
	while (cacheid < BLOCK_SIZE && cacheid < M) 
	{ 
		tempx[cacheid] = x[cacheid];
		cacheid += blockDim.x;
	}
	__syncthreads();

	if (warpID < M)
	{
		int start = row[warpID];
		int end = row[warpID + 1];
		float sum = 0.0;
		for (int i = start + lanedID; i < end; i += 32)
		{
			float tag = (col[i] >= BLOCK_SIZE) ? x[col[i]] : tempx[col[i]];		//如果处在共享内存内，则从共享内存中取数
			sum += (value[i] * tag);
		}

		//归约
		sum += __shfl_down(sum, 16);
		sum += __shfl_down(sum, 8);
		sum += __shfl_down(sum, 4);
		sum += __shfl_down(sum, 2);
		sum += __shfl_down(sum, 1);

		if (lanedID == 0) y[warpID] = sum;
	}
}

//固定数量的线程处理一行的数据
__global__ void spmv_csr3(float* value, int* col, int* row, float* x, float* y, int threads_per_row)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	int ward_id = tid / threads_per_row;
	int lane_id = tid % threads_per_row;

	if (ward_id < M)
	{
		int start = row[ward_id];
		int end = row[ward_id + 1];
		float sum = 0.0;
		for (int i = start + lane_id; i < end; i += threads_per_row)
		{
			sum += (value[i] * x[col[i]]);
			
		}

		//归约
		int i = threads_per_row >> 1;
		while (i != 0)
		{
			sum += __shfl_down(sum, i);
			i >>= 1;
		}

		if (lane_id == 0) y[ward_id] = sum;
	}
}


//固定数量的线程处理一行的数据(共享内存版本)
__global__ void spmv_csr3_shared(float* value, int* col, int* row, float* x, float* y, int threads_per_row)
{

	__shared__ float tempx[BLOCK_SIZE];
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	int ward_id = tid / threads_per_row;
	int lane_id = tid % threads_per_row;

	int cacheid = threadIdx.x;
	while (cacheid < BLOCK_SIZE && cacheid < M)
	{
		tempx[cacheid] = x[cacheid];
		cacheid += blockDim.x;
	}
	__syncthreads();

	if (ward_id < M)
	{
		int start = row[ward_id];
		int end = row[ward_id + 1];
		float sum = 0.0;
		for (int i = start + lane_id; i < end; i += threads_per_row)
		{
			float tag = (col[i] >= BLOCK_SIZE) ? x[col[i]] : tempx[col[i]];		//如果处在共享内存内，则从共享内存中取数
			sum += (value[i] * tag);
		}

		//归约
		int i = threads_per_row >> 1;
		while (i != 0)
		{
			sum += __shfl_down(sum, i);
			i >>= 2;
		}

		if (lane_id == 0) y[ward_id] = sum;
	}
}
//打印向量
void printfMatrix(float* y, int m)
{
	for (int i = 0; i < M; i++)
	{
		printf("%3.1f ", y[i]);
	}
	printf("\n");
}

void getSize(int &grid, int &blocks, int n)
{
	if (n < 1024)
	{
		blocks = 32;
		grid = (n + blocks - 1) / (blocks);
	}
	else if (n < 1024 * 32)
	{
		blocks = 128;
		grid = (n + blocks - 1) / (blocks);
	}
	else if (n < 1024 * 512)
	{
		blocks = 512;
		grid = (n + blocks - 1) / (blocks);
	}
	else
	{
		blocks = 1024;
		grid = (n + blocks - 1) / (blocks);
	}
}
float TEMP[M][M];
int main()
{
	int threadsPerBlock, blocksPerGrid, threads_per_row;
	float elapsedTime1, elapsedTime2, elapsedTime3, elapsedTime4, elapsedTime5, elapsedTime6;
	//float* TEMP;;
	srand((unsigned)time(NULL));

	cudaEvent_t start1, stop1, start2, stop2, start3, stop3, start4, stop4, start5, stop5, start6, stop6;
	cudaEventCreate(&start1);
	cudaEventCreate(&stop1);
	cudaEventCreate(&start2);
	cudaEventCreate(&stop2);
	cudaEventCreate(&start3);
	cudaEventCreate(&stop3);
	cudaEventCreate(&start4);
	cudaEventCreate(&stop4);
	cudaEventCreate(&start5);
	cudaEventCreate(&stop5);
	cudaEventCreate(&start6);
	cudaEventCreate(&stop6);

	//创建稀疏矩阵
	int num = 0;
	//TEMP = (float*)malloc(sizeof(float) * M * M);
	for (int i = 0; i < M; i++)
	{
		for (int j = 0; j < M; j++)
		{
			if (rand() < 10000)
			{
				TEMP[i][j] = 1.0;
				num++;
			}
			else
			{
				TEMP[i][j] = 0.0;
			}
		}
	}

	//分配主机端内存
	float* val;
	int* col;
	int* row;
	float* x, * y;
	x = (float*)malloc(sizeof(float) * M);
	y = (float*)malloc(sizeof(float) * M);
	val = (float*)malloc(sizeof(float) * num);
	col = (int*)malloc(sizeof(int) * num);
	row = (int*)malloc(sizeof(int) * (M + 1));

	//稀疏矩阵转为CSR存储格式
	int c = 0;
	int t = 0;
	row[0] = 0;
	for (int i = 0; i < M; i++)
	{
		t = 0;
		for (int j = 0; j < M; j++)
		{
			if (TEMP[i][j] != 0)
			{
				val[c] = TEMP[i][j];
				col[c] = j;
				c++;
				t++;
			}
		}
		y[i] = 0.0;
		x[i] = 1.0;
		row[i + 1] = row[i] + t;
	}

	//分配设备端内存
	float* dev_val, *dev_x, *dev_y;
	int* dev_col;
	int* dev_row;
	int size1 = (M + 1) * sizeof(int);
	HANDLE_ERROR(cudaMalloc((void**)&dev_val, sizeof(float) * num));
	HANDLE_ERROR(cudaMalloc((void**)&dev_row, size1));
	HANDLE_ERROR(cudaMalloc((void**)&dev_col, sizeof(int) * num));
	HANDLE_ERROR(cudaMalloc((void**)&dev_x, sizeof(float) * M));
	HANDLE_ERROR(cudaMalloc((void**)&dev_y, sizeof(float) * M));
	
	HANDLE_ERROR(cudaMemcpy(dev_val, val, num * sizeof(float), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_col, col, num * sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_row, row, (M+1) * sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_x, x, M * sizeof(float), cudaMemcpyHostToDevice));


	clock_t start_t, end_t;
	double total_t;
	//spmv_cpu方法的计算和性能分析
	//int start_time = (unsigned)time(NULL);
	start_t = clock();
	spmv_cpu(val, col, row, x, y);
	//int end_time = (unsigned)time(NULL);
	end_t = clock();
	total_t = (double)(end_t - start_t) * 1000 / CLOCKS_PER_SEC;
	printf("spmv_cpu spend time is %3.1f ms.\n", total_t);


	//spmv_csr1方法的计算和性能分析
	HANDLE_ERROR(cudaEventRecord(start1, 0));

	getSize(blocksPerGrid, threadsPerBlock, M);
	spmv_csr1 << < blocksPerGrid, threadsPerBlock >> > (dev_val, dev_col, dev_row, dev_x, dev_y);
	HANDLE_ERROR(cudaEventRecord(stop1, 0));
	HANDLE_ERROR(cudaEventSynchronize(stop1));
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime1, start1, stop1));
	//printfMatrix(y, M);
	printf("spmv_csr1 spend time is %3.1f ms.\n", elapsedTime1);


	//spmv_csr2方法的计算和性能分析
	HANDLE_ERROR(cudaEventRecord(start2, 0));
    threads_per_row = 32;

	getSize(blocksPerGrid, threadsPerBlock, M* threads_per_row);
	spmv_csr2 << < blocksPerGrid, threadsPerBlock >> > (dev_val, dev_col, dev_row, dev_x, dev_y);
	HANDLE_ERROR(cudaEventRecord(stop2, 0));
	HANDLE_ERROR(cudaEventSynchronize(stop2));
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime2, start2, stop2));
	cudaMemcpy(y, dev_y, M * sizeof(float), cudaMemcpyDeviceToHost);
	//printfMatrix(y, M);
	printf("spmv_csr2 spend time is %3.1f ms.\n", elapsedTime2);

	//spmv_csr3方法的计算和性能分析
	cudaEventRecord(start3, 0);
	threads_per_row = 16;
	getSize(blocksPerGrid, threadsPerBlock, M * threads_per_row);
	spmv_csr3 << < blocksPerGrid, threadsPerBlock >> > (dev_val, dev_col, dev_row, dev_x, dev_y, threads_per_row);
	cudaEventRecord(stop3, 0);
	cudaEventSynchronize(stop3);
	cudaEventElapsedTime(&elapsedTime3, start3, stop3);
	cudaMemcpy(y, dev_y, M * sizeof(float), cudaMemcpyDeviceToHost);
	//printfMatrix(C, N, N);
	printf("spmv_csr3 spend time is %3.1f ms.\n", elapsedTime3);


	cudaEventRecord(start4, 0);
	getSize(blocksPerGrid, threadsPerBlock, M);
	spmv_csr1_shared << < blocksPerGrid, threadsPerBlock >> > (dev_val, dev_col, dev_row, dev_x, dev_y);
	cudaEventRecord(stop4, 0);
	cudaEventSynchronize(stop4);
	cudaEventElapsedTime(&elapsedTime4, start4, stop4);
	cudaMemcpy(y, dev_y, M * sizeof(float), cudaMemcpyDeviceToHost);

	printf("spmv_csr1_share spend time is %3.1f ms.\n", elapsedTime4);


	//spmv_csr2方法的计算和性能分析
	cudaEventRecord(start5, 0);
	threads_per_row = 32;
	getSize(blocksPerGrid, threadsPerBlock, M * threads_per_row);
	spmv_csr2_shared << < blocksPerGrid, threadsPerBlock >> > (dev_val, dev_col, dev_row, dev_x, dev_y);
	cudaEventRecord(stop5, 0);
	cudaEventSynchronize(stop5);
	cudaEventElapsedTime(&elapsedTime5, start5, stop5);
	cudaMemcpy(y, dev_y, M * sizeof(float), cudaMemcpyDeviceToHost);
	//printfMatrix(C, N, N);
	printf("spmv_csr2_shared spend time is %3.1f ms.\n", elapsedTime5);

	//spmv_csr3_shared方法的计算和性能分析
	cudaEventRecord(start6, 0);
	threads_per_row = 16;
	getSize(blocksPerGrid, threadsPerBlock, M * threads_per_row);
	spmv_csr3_shared << < blocksPerGrid, threadsPerBlock >> > (dev_val, dev_col, dev_row, dev_x, dev_y, threads_per_row);
	cudaEventRecord(stop6, 0);
	cudaEventSynchronize(stop6);
	cudaEventElapsedTime(&elapsedTime6, start6, stop6);
	cudaMemcpy(y, dev_y, M * sizeof(float), cudaMemcpyDeviceToHost);
	//printfMatrix(C, N, N);
	printf("spmv_csr3_shared spend time is %3.1f ms.\n", elapsedTime6);

	return 0;
}