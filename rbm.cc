#include <cassert>
#include <cstdio>
#include <cstdlib>

#include <iostream>
#include <fstream>
#include <sstream>

#include <string>

#include <sys/stat.h>
#include <ctime>
#include <cmath>
#include <omp.h>




#include <cuda_runtime.h>
#include <cuda.h>
#include <curand.h>
#include "rbm_cuda.cuh"



#define BATCH_SIZE (16 * 512)

#define maxBlocks 256
#define threadsPerBlock 512




#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
	printf("Error at %s:%d\n",__FILE__,__LINE__); \
	exit(EXIT_FAILURE);}} while(0) 

#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
	printf("Error at %s:%d\n",__FILE__,__LINE__); \
	exit(EXIT_FAILURE);}} while(0)



using namespace std;




class record {

public:
	unsigned int user;
	unsigned int movie;
	unsigned int date;
	float score;
};


ostream & operator << (ostream &output, record& rcd) {
	output << rcd.user << " " << rcd.movie << " " << rcd.date << " " << rcd.score;
	return output;
}
istream & operator >> (istream &input, record& rcd) {
	input >> rcd.user >> rcd.movie >> rcd.date >> rcd.score;
	return input;
}


class record_array {
public:
	record * data;
	int size;

	record_array() {
		data = NULL;
		size = 0;
	}

	record_array(const char *file_name) {
		data = NULL;
		size = 0;
		load(file_name);
	}

	~record_array() {
		if (data != NULL) {
			delete [] data;
			data = NULL;
		}
	}

	const record & operator [] (unsigned int index) const{
		return data[index];
	}

	void load(const char *file_name) {
		if (data != NULL) {
			delete data;
			data = NULL;
		}

		struct stat results;
		// Get the size of the file
		if (stat(file_name, &results) != 0) {
			cout << "Stat Error" << endl;
			// system("pause");
			exit(-1);
		}

		// Get the size of the array
		size = results.st_size / sizeof(record);

		// Allocate memory
		data = new record[size];

		ifstream data_from_file;
		data_from_file.open(file_name, ios::binary | ios::in);

		data_from_file.read((char *) data, size * sizeof(record));
	}
};

float RMSE(const int* test_ratings, const float* predictions, const int test_data_size) {
	float s = 0;
	for (int i = 0; i < test_data_size; i++) {
		// TEST CODE
		if (predictions[i] > 5 || predictions[i] < 1) {
			cout << "bad input at " << i << " with pre = " << predictions[i] << endl;
			exit(-1);
		}
		s += (test_ratings[i] - predictions[i]) * (test_ratings[i] - predictions[i]);
	}
	return sqrt(s / test_data_size);
}




class RBM {
public:
	unsigned int N;
	unsigned int M;
	unsigned int CD_K;
	float lrate; // learning rate
	float lrate_BH;


	int* train_vec; // num_users * (start, end)
	int* test_vec;
	int* qual_vec;

	float* dev_W;
	float* dev_BV;
	float* dev_BH;

	float* dev_W_inc;
	float* dev_BV_inc;
	float* dev_BH_inc;

	// allocate memory for V0, Vt, H0, Ht ...
	float* dev_Vzeros;
	float* dev_Vts;
	float* dev_Hzeros;
	float* dev_Hts;


	float* dev_results; // num_records

	int* train_movies;
	int* train_ratings;

	int* test_movies;
	int* test_ratings;
	int test_data_size;

	unsigned int MAX_NUM_MOVIES_IN_BATCH;
	curandGenerator_t gen;


	cudaEvent_t timer_start, timer_stop;


	RBM(string train_file_name, string test_file_name, string qual_file_name) {


		record_array train_data;
		record_array test_data;
		record_array qual_data;

		train_data.load(train_file_name.c_str());
		test_data.load(test_file_name.c_str());
		qual_data.load(qual_file_name.c_str());
		cout << "finish loading " << qual_file_name << endl;


		// Setting up train movies & ratings & N & M
		train_movies = new int[train_data.size];
		train_ratings = new int[train_data.size];

		setNumUserMovieOthers(train_data, &N, &M, &MAX_NUM_MOVIES_IN_BATCH, train_movies, train_ratings);
		cout << "train: " << train_data.size << "\ttest: " << test_data.size << "\tqual: " << qual_data.size << endl;
		cout << "users: " << N << "\tmovies: " << M << "\tMAX_NUM_MOVIES_IN_BATCH: " << MAX_NUM_MOVIES_IN_BATCH << endl;

		// Setting up test movies & ratings & N & M
		unsigned int tmp1, tmp2, tmpMax;
		test_movies = new int[test_data.size];
		test_ratings = new int[test_data.size];
		test_data_size = test_data.size;
		setNumUserMovieOthers(test_data, &tmp1, &tmp2, &tmpMax, test_movies, test_ratings);
		MAX_NUM_MOVIES_IN_BATCH = max(MAX_NUM_MOVIES_IN_BATCH, tmpMax);




		CD_K = 1;
		// lrate = 0.001;
		lrate = 0.1 * M / train_data.size;
		lrate_BH = lrate / BATCH_SIZE;


		// The even out on M is because of the bug on Nvidia Curand function
		M = (M + 1) / 2 * 2;
		cout << "after evening out, M = " << M << endl;



		train_vec = make_pre_vec(train_data, N);
		test_vec = make_pre_vec(test_data, N);
		qual_vec = make_pre_vec(qual_data, N);


		cout << "finish making pre vecs" << endl;


		// Create a pseudo-random number generator
		const float mean = 0.0;
		const float std_dev = 0.08;
		CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
		CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, (unsigned long long) clock()));

		CUDA_CALL(cudaMalloc(&dev_W, sizeof(float) * K * F * M));
		CUDA_CALL(cudaMalloc(&dev_BV, sizeof(float) * K * M));
		CUDA_CALL(cudaMalloc(&dev_BH, sizeof(float) * F));

		CURAND_CALL(curandGenerateNormal(gen, dev_W, K * F * M, mean, std_dev));
		CURAND_CALL(curandGenerateNormal(gen, dev_BV, K * M, mean, std_dev));
		CURAND_CALL(curandGenerateNormal(gen, dev_BH, F, mean, std_dev));

		CUDA_CALL(cudaMalloc(&dev_W_inc, sizeof(float) * K * F * M));
		CUDA_CALL(cudaMalloc(&dev_BV_inc, sizeof(float) * K * M));
		CUDA_CALL(cudaMalloc(&dev_BH_inc, sizeof(float) * F * BATCH_SIZE));


		CUDA_CALL(cudaMalloc(&dev_Vzeros, sizeof(float) * K * MAX_NUM_MOVIES_IN_BATCH));
		CUDA_CALL(cudaMalloc(&dev_Vts, sizeof(float) * K * MAX_NUM_MOVIES_IN_BATCH));
		CUDA_CALL(cudaMalloc(&dev_Hzeros, sizeof(float) * F * BATCH_SIZE));
		CUDA_CALL(cudaMalloc(&dev_Hts, sizeof(float) * F * BATCH_SIZE));


		cout << "finish pre-processing" << endl;
	}

	~RBM() {

		delete[] train_vec;
		delete[] test_vec;
		delete[] qual_vec;


		cudaFree(dev_W);
		cudaFree(dev_BV);
		cudaFree(dev_BH);

		cudaFree(dev_W_inc);
		cudaFree(dev_BV_inc);
		cudaFree(dev_BH_inc);

		cudaFree(dev_Vzeros);
		cudaFree(dev_Vts);
		cudaFree(dev_Hzeros);
		cudaFree(dev_Hts);

		
		cudaFree(dev_results);

		delete[] train_movies;
		delete[] train_ratings;

		delete[] test_movies;
		delete[] test_ratings;

		curandDestroyGenerator(gen);

	}

	void fit(unsigned int n_iter = 1) {

		// setup recording
		float predict_timer = 0;
		float train_timer = 0;
		float milliseconds;
		cudaEvent_t timer_start, timer_stop;
		cudaEventCreate(&timer_start);
		cudaEventCreate(&timer_stop);


		// TEST CODE Memory Allocation Area
		float* results = new float[test_data_size];
		float* W = new float[10];
		float* BV = new float[10];
		float* BH = new float[10];
		float* H = new float[10];
		float* V = new float[10];
		float* BV_inc = new float[10];


		// allocate memory to store movies and ratings
		int* dev_train_vec_in_batch;
		int* dev_movies_in_batch;
		int* dev_ratings_in_batch;



		CUDA_CALL(cudaMalloc(&dev_train_vec_in_batch, sizeof(int) * 2 * BATCH_SIZE));
		CUDA_CALL(cudaMalloc(&dev_movies_in_batch, sizeof(int) * MAX_NUM_MOVIES_IN_BATCH));
		CUDA_CALL(cudaMalloc(&dev_ratings_in_batch, sizeof(int) * MAX_NUM_MOVIES_IN_BATCH));




		CUDA_CALL(cudaMalloc(&dev_results, sizeof(float) * test_data_size));


		int* dev_test_vec_in_batch;
		int* dev_test_movies_in_batch;
		int* dev_test_ratings_in_batch;
		CUDA_CALL(cudaMalloc(&dev_test_vec_in_batch, sizeof(int) * 2 * BATCH_SIZE));
		CUDA_CALL(cudaMalloc(&dev_test_movies_in_batch, sizeof(int) * MAX_NUM_MOVIES_IN_BATCH));
		CUDA_CALL(cudaMalloc(&dev_test_ratings_in_batch, sizeof(int) * MAX_NUM_MOVIES_IN_BATCH));







		for (unsigned int iter_num = 0; iter_num < n_iter; iter_num++) {

			cudaEventRecord(timer_start, 0);

			cout << "iteration: " << iter_num << endl; 

			// // make prediction

			for (unsigned int iv = 0; iv < N; iv += BATCH_SIZE) {
				int batch_size = min((unsigned int) BATCH_SIZE, N - iv);

				// For train
				int i_batch_start = train_vec[2 * iv];
				int i_batch_end = train_vec[2 * (iv + batch_size) - 1];
				int num_movies_in_this_batch = i_batch_end - i_batch_start;

				// For test
				int i_test_batch_start = test_vec[2 * iv];
				int i_test_batch_end = test_vec[2 * (iv + batch_size) - 1];
				int num_test_movies_in_this_batch = i_test_batch_end - i_test_batch_start;


				cudaMemcpy(dev_train_vec_in_batch, train_vec + 2 * iv, sizeof(int) * 2 * batch_size, cudaMemcpyHostToDevice);
				cudaMemcpy(dev_movies_in_batch, train_movies + i_batch_start, sizeof(int) * num_movies_in_this_batch, cudaMemcpyHostToDevice);
				cudaMemcpy(dev_ratings_in_batch, train_ratings + i_batch_start, sizeof(int) * num_movies_in_this_batch, cudaMemcpyHostToDevice);

				cudaMemcpy(dev_test_vec_in_batch, test_vec + 2 * iv, sizeof(int) * 2 * batch_size, cudaMemcpyHostToDevice);
				cudaMemcpy(dev_test_movies_in_batch, test_movies + i_test_batch_start, sizeof(int) * num_test_movies_in_this_batch, cudaMemcpyHostToDevice);
				cudaMemcpy(dev_test_ratings_in_batch, test_ratings + i_test_batch_start, sizeof(int) * num_test_movies_in_this_batch, cudaMemcpyHostToDevice);


				cudaMemset(dev_Vzeros, 0, sizeof(float) * K * num_movies_in_this_batch);
				cudaMemset(dev_Hzeros, 0, sizeof(float) * F * batch_size);

				cudaMemset(dev_Vts, 0, sizeof(float) * K * num_test_movies_in_this_batch);
				cudaMemset(dev_Hts, 0, sizeof(float) * F * batch_size);

				int blocks = min(maxBlocks, (int) ceil(
					batch_size / (float)threadsPerBlock));

				float* dev_results_in_batch = dev_results + i_test_batch_start;

				train(dev_train_vec_in_batch, dev_movies_in_batch, dev_ratings_in_batch,
					dev_test_vec_in_batch, dev_test_movies_in_batch, dev_test_ratings_in_batch,
					dev_Vzeros, dev_Vts, dev_Hzeros, dev_Hts,
					dev_W, dev_BV, dev_BH, dev_W_inc, dev_BV_inc, dev_BH_inc,
					batch_size, num_movies_in_this_batch, i_batch_start,
					num_test_movies_in_this_batch, i_test_batch_start,
					M, lrate, lrate_BH,
					dev_results_in_batch, false,
					blocks, threadsPerBlock);


				// TEST CODE
				// if (iv % 5 == 0) {
				// 	cudaMemcpy(H, dev_Hzeros + 20, sizeof(float) * 10, cudaMemcpyDeviceToHost);
				// 	for (unsigned int i = 0; i < 10; i++)
				// 		cout << H[i] << " ";
				// 	cout << endl;

				// 	cudaMemcpy(V, dev_Vts + 20, sizeof(float) * 10, cudaMemcpyDeviceToHost);
				// 	for (unsigned int i = 0; i < 10; i++)
				// 		cout << V[i] << " ";
				// 	cout << endl;
				// }
			}

			cudaEventRecord(timer_stop, 0);
			cudaEventSynchronize(timer_stop);
			cudaEventElapsedTime(&milliseconds, timer_start, timer_stop);
			cout << "time spent on predicting = " << milliseconds << "\t" << flush;
			predict_timer += milliseconds;


			// compute RMSE:
			cudaMemcpy(results, dev_results, sizeof(float) * test_data_size, cudaMemcpyDeviceToHost);
			float rmse = RMSE(test_ratings, results, test_data_size);
			cout << "RMSE = " << rmse << endl;





			// start timer to record training and updating process
			cudaEventRecord(timer_start, 0);


			// update weights
			for (unsigned int iv = 0; iv < N; iv += BATCH_SIZE) {
				// if (iv % 3 == 0)
				// 	cout << "." << flush;


				int batch_size = min((unsigned int) BATCH_SIZE, N - iv);
				int i_batch_start = train_vec[2 * iv];
				int i_batch_end = train_vec[2 * (iv + batch_size) - 1];
				int num_movies_in_this_batch = i_batch_end - i_batch_start;


				cudaMemset(dev_W_inc, 0, sizeof(float) * K * F * M);
				cudaMemset(dev_BV_inc, 0, sizeof(float) * K * M);
				cudaMemset(dev_BH_inc, 0, sizeof(float) * F * BATCH_SIZE);


				cudaMemcpy(dev_train_vec_in_batch, train_vec + 2 * iv, sizeof(int) * 2 * batch_size, cudaMemcpyHostToDevice);
				cudaMemcpy(dev_movies_in_batch, train_movies + i_batch_start, sizeof(int) * num_movies_in_this_batch, cudaMemcpyHostToDevice);
				cudaMemcpy(dev_ratings_in_batch, train_ratings + i_batch_start, sizeof(int) * num_movies_in_this_batch, cudaMemcpyHostToDevice);


				cudaMemset(dev_Vzeros, 0, sizeof(float) * K * num_movies_in_this_batch);
				cudaMemset(dev_Vts, 0, sizeof(float) * K * num_movies_in_this_batch);
				cudaMemset(dev_Hzeros, 0, sizeof(float) * F * batch_size);
				cudaMemset(dev_Hts, 0, sizeof(float) * F * batch_size);
				// cudaMemcpy(dev_Vts, dev_BV, sizeof(float) * K * num_movies_in_this_batch, cudaMemcpyHostToDevice);
				// cudaMemcpy(dev_Hzeros, dev_BH, sizeof(float) * F * batch_size, cudaMemcpyHostToDevice);
				// cudaMemcpy(dev_Hts, dev_BH, sizeof(float) * F * batch_size, cudaMemcpyHostToDevice);


				// user, movie, rating
				// train_vec -> start, end
				// W, BH, BV
				// W_inc, BH_inc, BV_inc
				int blocks = min(maxBlocks, (int) ceil(
					batch_size / (float)threadsPerBlock));



				train(dev_train_vec_in_batch, dev_movies_in_batch, dev_ratings_in_batch,
					NULL, NULL, NULL,
					dev_Vzeros, dev_Vts, dev_Hzeros, dev_Hts,
					dev_W, dev_BV, dev_BH, dev_W_inc, dev_BV_inc, dev_BH_inc,
					batch_size, num_movies_in_this_batch, i_batch_start, 
					0, 0,
					M, lrate, lrate_BH,
					NULL, true,
					blocks, threadsPerBlock);




				// // TEST CODE: verify BV_inc
				// if (iv % 5 == 0) {

				// 	cout << "Test part: " << endl;
				// 	cudaMemcpy(BV, dev_BV + 280, sizeof(float) * 10, cudaMemcpyDeviceToHost);
				// 	for (unsigned int i = 0; i < 10; i++) {
				// 		cout << BV[i] << " ";
				// 	}
				// 	cout << endl;
					
				// 	cudaMemcpy(W, dev_W + 280, sizeof(float) * 10, cudaMemcpyDeviceToHost);
				// 	for (unsigned int i = 0; i < 10; i++) {
				// 		cout << W[i] << " ";
				// 	}
				// 	cout << endl;
					
				// 	cudaMemcpy(BH, dev_BH, sizeof(float) * 10, cudaMemcpyDeviceToHost);
				// 	for (unsigned int i = 0; i < 10; i++) {
				// 		cout << BH[i] << " ";
				// 	}
				// 	cout << endl;

				// 	cudaMemcpy(BV_inc, dev_BV_inc + 280, sizeof(float) * 10, cudaMemcpyDeviceToHost);
				// 	for (unsigned int i = 0; i < 10; i++) {
				// 		cout << BV_inc[i] << " ";
				// 	}
				// 	cout << endl;
				// }
			}

			cudaEventRecord(timer_stop, 0);
			cudaEventSynchronize(timer_stop);
			cudaEventElapsedTime(&milliseconds, timer_start, timer_stop);
			cout << "time spent on training = " << milliseconds << endl;
			train_timer	+= milliseconds;

		}



		// compute average time spent per iteration
		cout << "overall speed info per iteration: " << endl;
		cout << "averaged predicting time = " << predict_timer / n_iter << endl; 
		cout << "averaged training time = " << train_timer / n_iter << endl;




		// TEST CODE memory release area
		delete[] results;
		delete[] W;
		delete[] BV;
		delete[] BH;
		delete[] V;
		delete[] H;
		delete[] BV_inc;
		

		cudaFree(dev_train_vec_in_batch);
		cudaFree(dev_movies_in_batch);
		cudaFree(dev_ratings_in_batch);

		cudaFree(dev_test_vec_in_batch);
		cudaFree(dev_test_movies_in_batch);
		cudaFree(dev_test_ratings_in_batch);


	}


	void setNumUserMovieOthers(const record_array &rcd_array, unsigned int* ptr_N, unsigned int* ptr_M, unsigned int* ptr_MAX_NUM_MOVIES_IN_BATCH, int* movies, int* ratings) {

		unsigned int cur_user = rcd_array.data[0].user;
		int num_users = 0;
		int num_movies_in_this_batch = 0;

		int batch_num = 1;
		*ptr_MAX_NUM_MOVIES_IN_BATCH = 0;

		// TEST CODE
		unsigned int largest_movie_id = 0;
		unsigned int last_user = 0;

		for (int i = 0; i < rcd_array.size; i++) {
			record r = rcd_array.data[i];

			if (r.user != cur_user) {
				cur_user = r.user;
				num_users++;
			}

			movies[i] = r.movie - 1;
			ratings[i] = r.score;

			if ((int)cur_user > batch_num * BATCH_SIZE) {
				*ptr_MAX_NUM_MOVIES_IN_BATCH = max(*ptr_MAX_NUM_MOVIES_IN_BATCH, (unsigned int)num_movies_in_this_batch);
				num_movies_in_this_batch = 0;

				batch_num++;
			}
			num_movies_in_this_batch++;


			// TEST CODE
			if (r.movie > largest_movie_id) {
				largest_movie_id = r.movie;
			}

			if (last_user > r.user) {
				cout << "this dataset is incorrect" << endl;
				exit(-1);
			}
			last_user = r.user;

		}

		// *ptr_N = users_set.size();
		*ptr_N = rcd_array.data[rcd_array.size - 1].user;
		*ptr_M = largest_movie_id;
		*ptr_MAX_NUM_MOVIES_IN_BATCH = max(*ptr_MAX_NUM_MOVIES_IN_BATCH, (unsigned int)num_movies_in_this_batch);

		cout << "setNumUserMovies debug info:" << endl;
		cout << "num_users = " << num_users << endl;
	}


	int* make_pre_vec(const record_array &rcd_array, int N) {
		int* rcd_vec = new int[2 * N];

		int user_id = rcd_array.data[0].user - 1;
		int start = 0;
		int end = 0;

		for (int vec_idx = 0; vec_idx < N; vec_idx++) {
			rcd_vec[2 * vec_idx] = start;

			while ((user_id == vec_idx) && (end < rcd_array.size)) {
				end++;
				if (end != rcd_array.size)
					user_id = rcd_array.data[end].user - 1;
			}

			rcd_vec[2 * vec_idx + 1] = end;

			start = end;
		}

		// cout << "make_pre_vec debug info:" << endl;
		// cout << "last user in loop = " << user_id << endl;
		// cout << "last user in record = " << rcd_array.data[rcd_array.size - 1].user << endl;
		// cout << "rcd_array.size = " << rcd_array.size << endl;
		// cout << "last element in this pre_vec " << rcd_vec[2 * N - 1] << endl;
		return rcd_vec;
	}
};


























int main(int argc, char** argv) {

	unsigned int ITER_NUM = 40;

	string train_file_name = "data/mid_main.data";
	string test_file_name = "data/mid_prob.data";
	string qual_file_name = "data/mid_prob.data";

	// string train_file_name = "data/main_data.data";
	// string test_file_name = "data/prob_data.data";
	// string qual_file_name = "data/prob_data.data"; // TODO: change to qual in the future


	RBM rbm(train_file_name, test_file_name, qual_file_name);
	rbm.fit(ITER_NUM);



	if (argc == 2) {
	ifstream ifs(argv[1]);
	// stringstream buffer;
	// buffer << ifs.rdbuf();
	// cluster(buffer, k, batch_size);
	}
}




