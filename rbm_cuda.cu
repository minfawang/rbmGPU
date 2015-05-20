#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>
#include <cuda.h>
#include <cublas_v2.h>

#include "rbm_cuda.cuh"

// #include <iostream>
// using namespace std;

/*
 * Generate a vector of length N with random single-precision floating-point
 * values between 0 and 100.
 */
void generate_random_vector(int N, float **outX)
{
    int i;
    double rMax = (double)RAND_MAX;
    float *X = (float *)malloc(sizeof(float) * N);

    for (i = 0; i < N; i++)
    {
        int r = rand();
        double dr = (double)r;
        X[i] = (dr / rMax) * 100.0;
    }

    *outX = X;
}

/*
 * Generate a matrix with M rows and N columns in column-major order. The matrix
 * will be filled with random single-precision floating-point values between 0
 * and 100.
 */
void generate_random_dense_matrix(int M, int N, float **outA)
{
    int i, j;
    double rMax = (double)RAND_MAX;
    float *A = (float *)malloc(sizeof(float) * M * N);

    // For each column
    for (j = 0; j < N; j++)
    {
        // For each row
        for (i = 0; i < M; i++)
        {
            double dr = (double)rand();
            A[j * M + i] = (dr / rMax) * 100.0;
        }
    }

    *outA = A;
}

void cudaTest() {

	cublasHandle_t handle = 0;
	// create the cuBLAS handle
	cublasCreate(&handle);

	int M = 8;
	int N = 8;


	float *A;
	float *X;
	float *Y;
	float *dA;
	float *dX;
	float *dY;

	generate_random_dense_matrix(M, N, &A);
	generate_random_vector(N, &X);
	generate_random_vector(M, &Y);

	cudaMalloc((void**) &dA, sizeof(float) * M * N);
	cudaMalloc((void**) &dX, sizeof(float) * N);
	cudaMalloc((void**) &dY, sizeof(float) * M);

	cublasSetMatrix(M, N, sizeof(float), A, M, dA, M);


	// // // copy 2th row from A to dX
	// cublasSetVector(N, sizeof(float), dA+2, M, dX, 1);

	cublasSetVector(M, sizeof(float), dA + 2 * M, 1, dX, 1);
	// cublasSetVector(M, sizeof(float), A+2, 1, dX, 1);
	// cublasSetVector(M, sizeof(float), Y, 1, dY, 1);


	cudaMemcpy(X, dX, sizeof(float) * N, cudaMemcpyDeviceToHost);



	// for (int j = 0; j < N; j++) {
	// 	for (int i = 0; i < M; i++) {
	// 		cout << A[j*M + i] << " ";
	// 	}
	// 	cout << endl;
	// }

	// for (int j = 0; j < M; j++) {
	// 	cout << X[j] << endl;
	// }



	cublasDestroy(handle);

}


__global__
void trainKernel(int* users, int* movies, int* ratings, int* starts, int* sizes, 
	float* A, float* B, float* BV, float* BH, int F, int C,
	float* Vzeros, float* Vts, float* Hzeros, float* Hts, float* W_users,
	int batch_size, int CD_K, cublasHandle_t &handle) {

	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	while (index < batch_size) {
		// TODO: Write me
		int user = users[index]; // user id
		int start = starts[index]; // the start index of movies in the batch
		int size = sizes[index]; // number of movies for this user

		float* H0 = Hzeros + index * F; // dim = F
		float* Ht = Hts + index * F; // dim = F

		float* V0 = Vzeros + start * K; // dim = K * size
		float* Vt = Vts + start * K; // dim = K * size

		float* W_user = W_users + start * K * F;


		// from start to (start + size)
		int* uMovies = movies + start; // dim = size
		int* uRatings = ratings + start; // dim = size
		

		// set up V0 and Vt based on the input data.
		for (int i = 0; i < size; i++) {
			V0[i * K + uRatings[i] - 1] = 1;
			Vt[i * K + uRatings[i] - 1] = 1;
			
	
			int ldA = K;
			int ldB = C;
			int ldW = K;

			const float alf = 1;
			const float bet = 0;
			const float* alpha = &alf;
			const float* beta = &bet;

			cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, K, F, C, alpha, A + i * K * C, ldA, B, ldB, beta, W_user + i * K * F, ldW);

		}


		/*////////////////// set up H0 by V -> H //////////////////
		H0(j) = sigma( BH(j) + sum_ik ( W(k, j, r.movie) * V0(k, i) ))*/

		// H0 = BH;
		// for (int i = 0; i < size; i++) {

		// 	H0 += W_user.slice(i).t() * V0.col(i);
		// }
		// H0 = 1.0 / (1 + exp(-H0));
		for (int j = 0; j < F; j++) {
			H0[j] = BH[j];
			for (int i = 0; i < size; i++) {

			}
		}


		index += blockDim.x * gridDim.x;
	}
}


void train(int* users, int* movies, int* ratings, int* starts, int* sizes, 
	float* A, float* B, float* BV, float* BH, int F, int C,
	float* Vzeros, float* Vts, float* Hzeros, float* Hts, float* W_users,
	int batch_size, int CD_K) {

	int block_size = (batch_size < 512) ? batch_size : 512;
	int grid_size = (batch_size + block_size -1) / block_size;


	// create the cuBLAS handle
	cublasHandle_t handle = 0;
	cublasCreate(&handle);

	trainKernel<<<grid_size, block_size>>>(users, movies, ratings, starts, sizes, 
		A, B, BV, BH, F, C, Vzeros, Vts, Hzeros, Hts, W_users, batch_size, CD_K, handle); 

	cublasDestroy(handle);
}
























