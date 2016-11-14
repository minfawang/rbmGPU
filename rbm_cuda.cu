#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda.h>
#include "rbm_cuda.cuh"



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
