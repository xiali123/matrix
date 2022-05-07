#include <stdio.h>
#include <math.h>
#include <time.h>

#define M 1024
#define NUM 8000
#define imin(a,b) (a>b)?b:a;
typedef float ElemType;
typedef long long ll;

//每个线程负责Cij的一个元素的计算
__global__ void spsemm_1(ElemType* value, int* col, int* row, ElemType* B, ElemType *C, int m, int n)
{
	int xx = threadIdx.x + blockIdx.x * blockDim.x;
	int yy = threadIdx.y + blockIdx.y * blockDim.y;
	int zz = threadIdx.z + blockIdx.z * blockDim.z;
	int tid = xx + yy * blockDim.x * gridDim.x + zz * blockDim.x * gridDim.x * blockDim.y * gridDim.y;

	//数组C的行列
	int bx = tid / m;
	int by = tid % m;

	if (bx < m)
	{
		int start = row[bx];
		int end = row[bx + 1];
		ElemType sum = 0.0;
		for (int i = start; i < end; i++)
		{
			
			sum += value[i] * B[col[i] * n + by];
			for (int i = 0; i < NUM; i++) float tag = 999999 * 138882;
		}

		C[bx * m + by] = sum;
	}
}

//固定数量的线程负责Cij一个线程的计算操作Gi
__global__ void spsemm_2(ElemType* value, int* col, int* row, ElemType* B, ElemType* C, int m, int n, int threads_per_row)
{
	int xx = threadIdx.x + blockIdx.x * blockDim.x;
	int yy = threadIdx.y + blockIdx.y * blockDim.y;
	int zz = threadIdx.z + blockIdx.z * blockDim.z;
	int tid = xx + yy * blockDim.x * gridDim.x + zz * blockDim.x * gridDim.x * blockDim.y * gridDim.y;

	int wid = tid / threads_per_row;
	int lane = tid / threads_per_row;

	//当前线程负责计算的数组C的元素行列
	int bx = wid / m;
	int by = wid % m;

	if (bx < m)
	{
		int start = row[bx];
		int end = row[bx + 1];
		ElemType sum = 0.0;
		for (int i = start; i < end; i+= threads_per_row)
		{
			sum += value[i] * B[col[i] * n + by];
			for (int i = 0; i < NUM; i++) float tag = 999999 * 138882;
		}

		//归约
		int i = threads_per_row / 2;
		while (i != 0)
		{
			sum += __shfl_down(sum, i);
			i /= 2;
		}

		if(lane == 0) C[bx * m + by] = sum;
	}
}

//一个线程束的线程负责Cij一个线程的计算操作
__global__ void spsemm_3(ElemType* value, int* col, int* row, ElemType* B, ElemType* C, int m, int n, int threads_per_row)
{
	int xx = threadIdx.x + blockIdx.x * blockDim.x;
	int yy = threadIdx.y + blockIdx.y * blockDim.y;
	int zz = threadIdx.z + blockIdx.z * blockDim.z;
	int tid = xx + yy * blockDim.x * gridDim.x + zz * blockDim.x * gridDim.x * blockDim.y * gridDim.y;

	int wid = tid / threads_per_row;
	int lane = tid / threads_per_row;


	int bx = wid / m;
	int by = wid % m;

	if (bx < m)
	{
		int start = row[bx];
		int end = row[bx + 1];
		ElemType sum = 0.0;
		for (int i = start; i < end; i += threads_per_row)
		{
			sum += value[i] * B[col[i] * n + by];
			for (int i = 0; i < NUM; i++) float tag = 999999 * 138882;
		}

		sum += __shfl_down(sum, 16);
		sum += __shfl_down(sum, 8);
		sum += __shfl_down(sum, 4);
		sum += __shfl_down(sum, 2);
		sum += __shfl_down(sum, 1);

		if (lane == 0) C[bx * m + by] = sum;
	}
}

//打印数组C
void printfMatrix(float* C, int m, int n)
{

	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			printf("%3.1f ", C[i * n + j]);
		}
		printf("\n");
	}
}

void getSize(int& grid, int& blocks, ll n)
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
	int blocksPerGrid ,threadsPerBlock;
	float elapsedTime;
	srand((unsigned)time(NULL));
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//创建稀疏矩阵
	int num = 0;
	for (int i = 0; i < M; i++)
	{
		for (int j = 0; j < M; j++)
		{
			if (rand() < 300)
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
	float *B, *C;

	B = (float*)malloc(sizeof(float) * M * M);
	C = (float*)malloc(sizeof(float) * M * M);
	val = (float*)malloc(sizeof(float) * num);
	col = (int*)malloc(sizeof(int) * num);
	row = (int*)malloc(sizeof(int) * (M + 1));

	//更改稀疏矩阵的结构为CSR
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
			B[i * M + j] = 1.0;
			C[i * M + j] = 1.0;
		}
		row[i + 1] = row[i] + t;
	}

	//分配设备端内存
	float* dev_val, * dev_B, * dev_C;
	int* dev_col;
	int* dev_row;

	cudaMalloc((void**)&dev_val, sizeof(float) * num);
	cudaMalloc((void**)&dev_row, sizeof(int) * (M + 1));
	cudaMalloc((void**)&dev_col, sizeof(int) * num);
	cudaMalloc((void**)&dev_B, sizeof(float) * M * M);
	cudaMalloc((void**)&dev_C, sizeof(float) * M* M);

	cudaMemcpy(dev_val, val, num * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_col, col, num * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_row, row, (M + 1) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_B, B, M * M * sizeof(float), cudaMemcpyHostToDevice);

	//spsemm_1方法的计算和性能分析
	cudaEventRecord(start, 0);
	
	getSize(blocksPerGrid, threadsPerBlock, M*M);
	spsemm_1 << < blocksPerGrid, threadsPerBlock >> > (dev_val, dev_col, dev_row, dev_B, dev_C, M, M );
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cudaMemcpy(C, dev_C, M * sizeof(float), cudaMemcpyDeviceToHost);
	//printfMatrix(C, M, M);
	printf("spsemm_1 spend time is %3.1f ms.\n", elapsedTime);


	//spsemm_2方法的计算和性能分析
	cudaEventRecord(start);
	int threads_per_row = 4;
	getSize(blocksPerGrid, threadsPerBlock, (ll)M * M * threads_per_row);
	spsemm_2 << <blocksPerGrid, threadsPerBlock >> > (dev_val, dev_col, dev_row, dev_B, dev_C, M, M ,threads_per_row);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cudaMemcpy(C, dev_C, M * sizeof(float), cudaMemcpyDeviceToHost);
	//printfMatrix(C, N, N);
	printf("spsemm_2 spend time is %3.1f ms.\n", elapsedTime);


	//spsemm_3方法的计算和性能分析
	cudaEventRecord(start);
	threads_per_row = 32;
	getSize(blocksPerGrid, threadsPerBlock, (ll)M * M * threads_per_row);
	spsemm_3 << <blocksPerGrid, threadsPerBlock >> > (dev_val, dev_col, dev_row, dev_B, dev_C,M,M, threads_per_row);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cudaMemcpy(C, dev_C, M * sizeof(float), cudaMemcpyDeviceToHost);
	//printfMatrix(C, N, N);
	printf("spsemm_3 spend time is %3.1f ms.\n", elapsedTime);
	return 0;
}