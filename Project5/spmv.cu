#include "common/book.h"

typedef float ElemType;

typedef struct
{
	ElemType* value;
	int* row_num;
	int* col_idx;
}matrix_csr;


__global__ void spmv(matrix_csr* mtx, ElemType* x, ElemType* y)
{
	ElemType* val = mtx->value;
	int* col = mtx->col_idx;
	int* row = mtx->row_num;

	int tid = threadIdx.x + 
}

int main()
{

	return 0;
}