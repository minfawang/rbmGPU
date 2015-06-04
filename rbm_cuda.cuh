#ifndef CUDA_RBM_CUH
#define CUDA_RBM_CUH

#define K 5
#define F 100

// void train(int* users, int* movies, int* ratings, int* starts, int* sizes, 
// 	float* A, float* B, float* B_old, float* BV, float* BH, int F, int C, float* lrate,
// 	float* Vzeros, float* Vts, float* Hzeros, float* Hts, float* W_users,
// 	int batch_size, int CD_K);x


void train(int* train_vec_in_batch, int* movies_in_batch, int* ratings_in_batch,
	int* test_vec_in_batch, int* test_movies_in_batch, int* test_ratings_in_batch,
	float* Vzeros, float* Vts, float* Hzeros, float* Hts,
	float* W, float* BV, float* BH, float* W_inc, float* BV_inc, float* BH_inc,
	int batch_size, int num_movies_in_this_batch, const int i_batch_start,
	const int num_test_movies_in_this_batch, const int i_test_batch_start,
	const unsigned int M, const float lrate, const float lrate_BH,
	float* results_in_batch, const bool update_weights,
	int blocks, int threadsPerBlock);

// void cudaTest();

#endif
