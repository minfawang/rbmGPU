#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>
#include <cuda.h>
// #include <cublas_v2.h>

#include "rbm_cuda.cuh"





// __device__ inline void floatAtomicAdd (float *address, float value)
//  {
//    int oldval, newval, readback;
 
//    oldval = __float_as_int(*address);
//    newval = __float_as_int(__int_as_float(oldval) + value);
//    while ((readback=atomicCAS((int *)address, oldval, newval)) != oldval) 
//      {
//       oldval = readback;
//       newval = __float_as_int(__int_as_float(oldval) + value);
//      }
//  }



// // #include <iostream>
// // using namespace std;

// /*
//  * Generate a vector of length N with random single-precision floating-point
//  * values between 0 and 100.
//  */
// void generate_random_vector(int N, float **outX)
// {
//     int i;
//     double rMax = (double)RAND_MAX;
//     float *X = (float *)malloc(sizeof(float) * N);

//     for (i = 0; i < N; i++)
//     {
//         int r = rand();
//         double dr = (double)r;
//         X[i] = (dr / rMax) * 100.0;
//     }

//     *outX = X;
// }

// /*
//  * Generate a matrix with M rows and N columns in column-major order. The matrix
//  * will be filled with random single-precision floating-point values between 0
//  * and 100.
//  */
// void generate_random_dense_matrix(int M, int N, float **outA)
// {
//     int i, j;
//     double rMax = (double)RAND_MAX;
//     float *A = (float *)malloc(sizeof(float) * M * N);

//     // For each column
//     for (j = 0; j < N; j++)
//     {
//         // For each row
//         for (i = 0; i < M; i++)
//         {
//             double dr = (double)rand();
//             A[j * M + i] = (dr / rMax) * 100.0;
//         }
//     }

//     *outA = A;
// }

// void cudaTest() {

// 	cublasHandle_t handle = 0;
// 	// create the cuBLAS handle
// 	cublasCreate(&handle);

// 	int M = 8;
// 	int N = 8;


// 	float *A;
// 	float *X;
// 	float *Y;
// 	float *dA;
// 	float *dX;
// 	float *dY;

// 	generate_random_dense_matrix(M, N, &A);
// 	generate_random_vector(N, &X);
// 	generate_random_vector(M, &Y);

// 	cudaMalloc((void**) &dA, sizeof(float) * M * N);
// 	cudaMalloc((void**) &dX, sizeof(float) * N);
// 	cudaMalloc((void**) &dY, sizeof(float) * M);

// 	cublasSetMatrix(M, N, sizeof(float), A, M, dA, M);


// 	// // // copy 2th row from A to dX
// 	// cublasSetVector(N, sizeof(float), dA+2, M, dX, 1);

// 	cublasSetVector(M, sizeof(float), dA + 2 * M, 1, dX, 1);
// 	// cublasSetVector(M, sizeof(float), A+2, 1, dX, 1);
// 	// cublasSetVector(M, sizeof(float), Y, 1, dY, 1);


// 	cudaMemcpy(X, dX, sizeof(float) * N, cudaMemcpyDeviceToHost);



// 	// for (int j = 0; j < N; j++) {
// 	// 	for (int i = 0; i < M; i++) {
// 	// 		cout << A[j*M + i] << " ";
// 	// 	}
// 	// 	cout << endl;
// 	// }

// 	// for (int j = 0; j < M; j++) {
// 	// 	cout << X[j] << endl;
// 	// }



// 	cublasDestroy(handle);

// }





__global__
void trainKernel(int* train_vec_in_batch, int* movies_in_batch, int* ratings_in_batch,
		float* Vzeros, float* Vts, float* Hzeros, float* Hts,
		float* W, float* BV, float* BH, float* W_inc, float* BV_inc, float* BH_inc,
		int batch_size, int num_movies_in_this_batch, const int i_batch_start,
		const bool update_weights) {

	unsigned int user = blockIdx.x * blockDim.x + threadIdx.x;
	while (user < batch_size) {
		int start = train_vec_in_batch[2 * user];
		int end = train_vec_in_batch[2 * user + 1];
		int size = end - start;

		start -= i_batch_start;


		if (size != 0) {
			float* V0 = Vzeros + K * start;
			float* Vt = Vts + K * start;
			float* H0 = Hzeros + user * F;
			float* Ht = Hts + user * F;
			
			int* u_movies = movies_in_batch + start;
			int* u_ratings = ratings_in_batch + start;

			// initialize V0
			for (int i = 0; i < size; i++) {
				V0[i * K + u_ratings[i] - 1] = 1;
			}

			//////////////// positive phase ////////////////
			for (int i = 0; i < size; i++) {
				float* W_user = W + u_movies[i] * (K * F);

				for (int j = 0; j < F; j++) {
					for (int k = 0; k < K; k++) {
						H0[j] += W_user[j * K + k] * V0[i * K + k];
					}
				}
			}

			// add bias and logistic function on H0
			for (int j = 0; j < F; j++) {
				H0[j] += BH[j];
				H0[j] = 1.0 / (1 + exp(-H0[j]));
			}



			if (update_weights) {
				//////////////// negative phase ////////////////
				for (int i = 0; i < size; i++) {
					float* W_user = W + u_movies[i] * (K * F);

					for (int j = 0; j < F; j++) {
						for (int k = 0; k < K; k++) {
							Vt[i * K + k] += H0[j] * W_user[j * K + k];
						}
					}

					// normalize Vt
					float sum_k = 0.0;
					for (int k = 0; k < K; k++) {

						Vt[i * K + k] += BV[u_movies[i] * K + k]; // add bias
						Vt[i * K + k] = exp(Vt[i * K + k]); // exponential
						sum_k += Vt[i * K + k];
					}
					for (int k = 0; k < K; k++) {
						Vt[i * K + k] /= sum_k;
					}
				}


				// compute Ht
				for (int i = 0; i < size; i++) {
					float* W_user = W + u_movies[i] * (K * F);

					for (int j = 0; j < F; j++) {
						for (int k = 0; k < K; k++) {
							Ht[j] += W_user[j * K + k] * Vt[i * K + k];
						}
					}
				}

				// add bias and logistic function on Ht
				for (int j = 0; j < F; j++) {
					Ht[j] += BV[j];
					Ht[j] = 1.0 / (1 + exp(-Ht[j]));
				}

				//////////////// update weight increments ////////////////
				// update BV_inc
				for (int i = 0; i < size; i++) {
					for (int k = 0; k < K; k++) {
						BV_inc[u_movies[i] * K + k] += (V0[i * K + k] - Vt[i * K + k]);
					}
				}

				// update W_inc
				for (int i = 0; i < size; i++) {
					for (int j = 0; j < F; j++) {
						for (int k = 0; k < K; k++) {
							W_inc[u_movies[i] * K * F + j * K + k] += (H0[j] * V0[i * K + k] - Ht[j] * Vt[i * K + k]);
						}
					}
				}

				// update BH_inc
				for (int j = 0; j < F; j++) {
					BH_inc[user * F + j] = (H0[j] - Ht[j]);
				}
				
			} // end update weights


		}

		user += blockDim.x * gridDim.x;
	}
}

__global__
void updateW_kernel(float* W, float* W_inc, const unsigned int M, const float lrate) {
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	while (i < K * F * M) {
		W[i] += lrate * W_inc[i];
		i += blockDim.x * gridDim.x;
	}
}

__global__
void updateBV_kernel(float* BV, float* BV_inc, const unsigned int M, const float lrate) {
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	while (i < K * M) {
		BV[i] += lrate * BV_inc[i];
		i += blockDim.x * gridDim.x;
	}
}

__global__
void updateBH_kernel(float* BH, float* BH_inc, const float lrate_BH, const int batch_size) {
	extern __shared__ float sBH_inc[];

	unsigned int tid = threadIdx.x;
	sBH_inc[tid] = 0;

	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	while (i < batch_size) {
		sBH_inc[tid] += BH_inc[i * F];

		i += blockDim.x * gridDim.x;
	}
	__syncthreads();

	for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
		if (tid < s) {
			sBH_inc[tid] += sBH_inc[tid + s];
		}
		__syncthreads();
	}

	if (tid < 32) {
		atomicAdd(&sBH_inc[tid], sBH_inc[tid + 32]);
		atomicAdd(&sBH_inc[tid], sBH_inc[tid + 16]);
		atomicAdd(&sBH_inc[tid], sBH_inc[tid + 8]);
		atomicAdd(&sBH_inc[tid], sBH_inc[tid + 4]);
		atomicAdd(&sBH_inc[tid], sBH_inc[tid + 2]);
		atomicAdd(&sBH_inc[tid], sBH_inc[tid + 1]);
	}

	if (tid == 0)
		atomicAdd(BH, lrate_BH * sBH_inc[0]);

}



__global__
void predictKernel(int* test_vec_in_batch, int* test_movies_in_batch, int* test_ratings_in_batch,
			float* Hzeros, float* Vts,
			float* W, float* BV,
			int batch_size, const int num_test_movies_in_this_batch, const int i_test_batch_start,
			float* results_in_batch) {

	unsigned int user = blockIdx.x * blockDim.x + threadIdx.x;

	while (user < batch_size) {
		int start = test_vec_in_batch[2 * user];
		int end = test_vec_in_batch[2 * user + 1];
		int size = end - start;

		start -= i_test_batch_start;


		if (size != 0) {
			float* H0 = Hzeros + user * F;
			float* Vt = Vts + K * start;
			
			int* u_movies = test_movies_in_batch + start;

			//////////////// negative phase ////////////////
			for (int i = 0; i < size; i++) {
				float* W_user = W + u_movies[i] * (K * F);

				for (int j = 0; j < F; j++) {
					for (int k = 0; k < K; k++) {
						Vt[i * K + k] += H0[j] * W_user[j * K + k];
					}
				}

				// normalize Vt
				float sum_k = 0.0;
				for (int k = 0; k < K; k++) {

					Vt[i * K + k] += BV[u_movies[i] * K + k]; // add bias
					Vt[i * K + k] = exp(Vt[i * K + k]); // exponential
					sum_k += Vt[i * K + k];
				}
				for (int k = 0; k < K; k++) {
					Vt[i * K + k] /= sum_k;
				}


				// update results
				float score = 0;
				for (int k = 0; k < K; k++) {
					score += (k + 1) * Vt[i * K + k];
				}
				results_in_batch[start + i] = score;
			}
		}

		user += blockDim.x * gridDim.x;
	}
}

void train(int* train_vec_in_batch, int* movies_in_batch, int* ratings_in_batch,
		int* test_vec_in_batch, int* test_movies_in_batch, int* test_ratings_in_batch,
		float* Vzeros, float* Vts, float* Hzeros, float* Hts,
		float* W, float* BV, float* BH, float* W_inc, float* BV_inc, float* BH_inc,
		int batch_size, int num_movies_in_this_batch, const int i_batch_start,
		const int num_test_movies_in_this_batch, const int i_test_batch_start,
		const unsigned int M, const float lrate, const float lrate_BH,
		float* results_in_batch, const bool update_weights,
		int blocks, int threadsPerBlock) {


	if (update_weights) {

		trainKernel<<<blocks, threadsPerBlock>>>
			(train_vec_in_batch, movies_in_batch, ratings_in_batch,
			Vzeros, Vts, Hzeros, Hts,
			W, BV, BH,
			W_inc, BV_inc, BH_inc,
			batch_size, num_movies_in_this_batch, i_batch_start,
			true);


		unsigned int Wblocks = min(blocks, (int)ceil(K * F * M / (float)threadsPerBlock));
		updateW_kernel<<<Wblocks, threadsPerBlock>>>(W, W_inc, M, lrate);

		unsigned int BVblocks = min(blocks, (int)ceil(K * M / (float)threadsPerBlock));
		updateBV_kernel<<<BVblocks, threadsPerBlock>>>(BV, BV_inc, M, lrate);

		unsigned int BHblocks = min(blocks, (int)ceil(batch_size / (float)threadsPerBlock));
		for (int j = 0; j < F; j++) {
			updateBH_kernel<<<BHblocks, threadsPerBlock, threadsPerBlock * sizeof(float)>>>
				(BH + j, BH_inc + j, lrate_BH, batch_size);
		}
	}

	// in prediction stage
	else {
		trainKernel<<<blocks, threadsPerBlock>>>
			(train_vec_in_batch, movies_in_batch, ratings_in_batch,
			Vzeros, Vts, Hzeros, Hts,
			W, BV, BH,
			W_inc, BV_inc, BH_inc,
			batch_size, num_movies_in_this_batch, i_batch_start,
			false);

		// TODO: update Vt, and compute results
		predictKernel<<<blocks, threadsPerBlock>>>
			(test_vec_in_batch, test_movies_in_batch, test_ratings_in_batch,
			Hzeros, Vts,
			W, BV,
			batch_size, num_test_movies_in_this_batch, i_test_batch_start,
			results_in_batch);
	}
}





/*
The code below is the unworked version using CUBLAS
*/


// __global__
// void trainKernel(int* users, int* movies, int* ratings, int* starts, int* sizes, 
// 	float* A, float* B, float* B_old, float* BV, float* BH, int F, int C, float* lrate,
// 	float* Vzeros, float* Vts, float* Hzeros, float* Hts, float* W_users,
// 	int batch_size, int CD_K, cublasHandle_t &handle) {


// 	// set up cuBlas multiplication parameters
// 	int ldA = K;
// 	int ldB = C;
// 	int ldW = K;
// 	int ldV = K;
// 	int ldH = F;

// 	const float MINUS = -1;
// 	const float ONE = 1;
// 	const float ZERO = 0;
// 	const float* pONE = &ONE;
// 	const float* pZERO = &ZERO;
// 	const float* pMINUS = &MINUS;

// 	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
// 	while (index < batch_size) {


// 		// TODO: Write me
// 		// int user = users[index]; // user id
// 		int start = starts[index]; // the start index of movies in the batch
// 		int size = sizes[index]; // number of movies for this user

// 		float* H0 = Hzeros + index * F; // dim = F * 1
// 		float* Ht = Hts + index * F; // dim = F * 1

// 		float* V0 = Vzeros + start * K; // dim = K * size
// 		float* Vt = Vts + start * K; // dim = K * size

// 		float* W_user = W_users + start * K * F; // dim = K * F * size


// 		// from start to (start + size)
// 		int* uMovies = movies + start; // dim = size
// 		int* uRatings = ratings + start; // dim = size

		

// 		// set up V0 and Vt based on the input data.
// 		for (int i = 0; i < size; i++) {
// 			V0[i * K + uRatings[i] - 1] = 1;
// 			Vt[i * K + uRatings[i] - 1] = 1;
			

// 			// Operation: W_user.slice(i) = A.slice(r.movie) * B;
// 			// W_user.slice(i) -> K * F
// 			// A.slice(r.movie) -> K * C
// 			// B -> C * F

// 			cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, K, F, C, pONE, A + uMovies[i] * K * C, ldA, B, ldB, pZERO, W_user + i * K * F, ldW);
// 		}


// 		/*////////////////// set up H0 by V -> H //////////////////
// 		H0(j) = sigma( BH(j) + sum_ik ( W(k, j, r.movie) * V0(k, i) ))*/

// 		// H0 = BH;
// 		// for (int i = 0; i < size; i++) {

// 		// 	H0 += W_user.slice(i).t() * V0.col(i);
// 		// }
// 		// H0 = 1.0 / (1 + exp(-H0));


		
// 		/*	W_user.slice(i).t() -> (K * F).t()
// 			V0.col(i) -> (F * 1)	*/


// 		// for (int j = 0; j < F; j++) {
// 		// 	H0[j] = BH[j];
// 		// }
// 		cublasScopy(handle, F, BH, 1, H0, 1);

// 		// Operation: H0 += W_user.slice(i).t() * V0.col(i) for i in range(size)
// 		// H0 -> F * 1
// 		// W_user(i).t() -> F * K
// 		// V0.col(i) -> K * 1
// 		for (int i = 0; i < size; i++) {
// 			cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, F, 1, K, pONE, W_user + i * K * F, ldW, V0 + i * K, ldV, pONE, H0, ldH);
// 		}
// 		// cublasSgemmBatched(handle, CUBLAS_OP_T, CUBLAS_OP_N, K, F, 1, pONE, (const float**)&W_user, ldW, (const float**)&V0, ldV, pONE, &H0, ldH, size);
// 		for (int j = 0; j < F; j++) {
// 			H0[j] = 1.0 / (1 + exp(-H0[j]));
// 		}




// 		/*
// 		/////////////////// Do the contrastive divergence ///////////////////
// 		for (int n = 0; n < CD_K; n++) {

// 			////////////// positive phase: V -> H /////////
// 			Ht = BH;
// 			for (int i = 0; i < size; i ++) {
// 				Ht += W_user.slice(i).t() * Vt.col(i);
// 			}
// 			Ht = 1.0 / (1 + exp(-Ht));
			

// 			// negative phase: H -> V
// 			for (int i = 0; i < size; i++) {
// 				Vt.col(i) = exp(BV.col(ims[i]) + W_user.slice(i) * Ht);
// 			}

// 			// Normalize Vt -> sum_k (Vt(k, i)) = 1
// 			Vt = normalise(Vt, 1);

// 		}
// 		*/

// 		/////////////////// Do the contrastive divergence ///////////////////
// 		for (int n = 0; n < CD_K; n++) {
// 			////////////// positive phase: V -> H /////////
// 			cublasScopy(handle, F, BH, 1, Ht, 1);
// 			// Operation: Ht += W_user.slice(i).t() * Vt.col(i);
// 			for (int i = 0; i < size; i++) {
// 				cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, F, 1, K, pONE, W_user + i * K * F, ldW, Vt + i * K, ldV, pONE, Ht, ldH);
// 			}
// 			// cublasSgemmBatched(handle, CUBLAS_OP_T, CUBLAS_OP_N, K, F, 1, pONE, (const float**)&W_user, ldW, (const float**)&Vt, ldV, pONE, &Ht, ldH, size);
// 			for (int j = 0; j < F; j++) {
// 				Ht[j] = 1.0 / (1 + exp(-Ht[j]));
// 			}

// 			// negative phase: H -> V
// 			for (int i = 0; i < size; i++) {
// 				// Operation: Vt.col(i) = exp(BV.col(ims[i]) + W_user.slice(i) * Ht);
// 				// Vt -> K * size
// 				// BV -> K * M
// 				// W_user -> K * F * size
// 				// Ht -> F * 1

// 				float sum_k = 0;
// 				cublasScopy(handle, K, BV + uMovies[i] * K, 1, Vt + i * K, 1);
// 				cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, K, 1, F, pONE, W_user + i * K * F, ldW, Ht, ldH, pZERO, Vt + i * K, ldV);
// 				for (int k = 0; k < K; k++) {
// 					float tmp = exp(*(Vt + i * K + k));
// 					*(Vt + i * K + k) = tmp;
// 					sum_k += tmp;
// 				}
// 				// Normalize Vt -> sum_k (Vt(k, i)) = 1
// 				for (int k = 0; k < K; k++) {
// 					*(Vt + i * K + k) /= sum_k;
// 				}
// 			}
// 		}



// 		// // TEST CODE
// 		// for (int k = 0; k < K; k++) {
// 		// 	printf("%f ", *(Vt + k));
// 		// }
// 		// printf("\n");



// 		/*
// 		// update BH
// 		BH += lrate * (H0 - Ht);
// 		*/

// 		// Operation: HV_diffs = V0.col(i) * H0.t() - Vt.col(i) * Ht.t() for i in range(size)
// 		// compute HV_diffs -> K * F * size
// 		// W_user -> K * F * size
// 		// V0 -> K * size
// 		// Vo.col() -> K * 1
// 		// H0 -> F * 1
// 		for (int i = 0; i < size; i++) {
// 			cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, K, F, 1, pONE, V0 + i * K, ldV, H0, ldH, pZERO, W_user + i * K * F, ldW);
// 			cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, K, F, 1, pMINUS, Vt + i * K, ldV, Ht, ldH, pONE, W_user + i * K * F, ldW);
// 		}

// 		// Operation: H_diffs = H0 - Ht
// 		cublasSaxpy(handle, F, pMINUS, Ht, 1, H0, 1);

// 		// Opeartion: V_diffs = V0.col(i) - Vt.col(i) for i in range(size)
// 		cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, K, size, pMINUS, Vt, ldV, pONE, V0, ldV, V0, ldV);
		
// 		// update BH
// 		// Operation: BH += lrate * (H0 - Ht);
// 		// BH -> F * 1
// 		cublasSaxpy(handle, F, lrate, H0, 1, BH, 1);



// 		/*
// 		// update B
// 		// update BV
// 		// update A
// 		mat B_old = B;
// 		for (int i = 0; i < size; i++) {
// 			mat HV_diff = (V0.col(i) * H0.t() - Vt.col(i) * Ht.t());
// 			A.slice(ims[i]) += lrate * HV_diff * B_old.t();
// 		}*/

// 		for (int i = 0; i < size; i++) {
// 			// update BV
// 			// Operation: BV.col(ims[i]) += lrate * (V0.col(i) - Vt.col(i))
// 			cublasSaxpy(handle, K, lrate, V0 + i * K, 1, BV + uMovies[i] * K, 1);

// 			// update B
// 			// Operation: B += lrate * A.slice(ims[i]).t() * HV_diff;
// 			// B -> C * F
// 			// A.slice(im[i]).t() -> (K * C).t() -> C * K
// 			// W_user(i) -> K * F
// 			cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, C, F, K, lrate, A + uMovies[i] * C * K, ldA, W_user + i * K * F, ldW, pONE, B, ldB);

// 			// update A
// 			// Operation: A.slice(ims[i]) += lrate * HV_diff * B_old.t();
// 			// A.slice(im[i]) -> K * C
// 			// W_user(i) -> K * F
// 			// B_old.t() -> F * C
// 			cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, K, C, F, lrate, W_user + i * K * F, ldW, B_old, ldB, pONE, A + uMovies[i] * K * C, ldA);

// 		}




// 		index += blockDim.x * gridDim.x;
// 	}
// }


// void train(int* users, int* movies, int* ratings, int* starts, int* sizes, 
// 	float* A, float* B, float* B_old, float* BV, float* BH, int F, int C, float* lrate,
// 	float* Vzeros, float* Vts, float* Hzeros, float* Hts, float* W_users,
// 	int batch_size, int CD_K) {

// 	int block_size = (batch_size < 512) ? batch_size : 512;
// 	int grid_size = (batch_size + block_size -1) / block_size;



// 	// create the cuBLAS handle
// 	cublasHandle_t handle = 0;
// 	cublasCreate(&handle);

// 	trainKernel<<<grid_size, block_size>>>(users, movies, ratings, starts, sizes, 
// 		A, B, B_old, BV, BH, F, C, lrate, Vzeros, Vts, Hzeros, Hts, W_users, batch_size, CD_K, handle); 



// 	cublasDestroy(handle);
// }
























