#ifndef CUDA_RBM_CUH
#define CUDA_RBM_CUH

#define K 5

void train(int* users, int* movies, int* ratings, int* starts, int* sizes, 
	float* A, float* B, float* BV, float* BH, int F, int C,
	float* Vzeros, float* Vts, float* Hzeros, float* Hts, float* W_users,
	int batch_size, int CD_K);


void cudaTest();

#endif
