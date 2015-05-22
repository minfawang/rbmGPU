#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <sstream>
#include <unordered_map>

#include <sys/stat.h>
#include <ctime>
#include <vector>
#include <omp.h>


#include <cuda_runtime.h>
#include <cuda.h>
#include <curand.h>
#include "rbm_cuda.cuh"



#define NUM_THREADS 8
#define BATCH_SIZE (NUM_THREADS * 256)

// TODO: change NUM_MOVIES_PER_BATCH when switching to large data
// number of movies per batch in mid_data: max = 97527, min = 86203
// in full data: max = 493495, min = 402065
#define NUM_MOVIES_PER_BATCH 97527


using namespace std;


void GPU_fill_rand(float* A, int nr_rows_A, int nr_cols_A) {
	// Create a pseudo-random number generator
	curandGenerator_t prng;
	curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);

	// Set the seed for the random number generator using the system clock
	curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());

	// Fill the array with random numbers on the device
	curandGenerateUniform(prng, A, nr_rows_A * nr_cols_A);
}


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

float RMSE(const record_array & test_data, const vector<float> & prediction) {
	double s = 0;
	for (int i = 0; i < test_data.size; i++) {
		s += (test_data[i].score - prediction[i]) * (test_data[i].score - prediction[i]);
	}
	return sqrt(s / test_data.size);
}




class RBM {
public:
	// cube A;
	// mat B;
	// mat BV;
	// vec BH;

	unsigned int C;
	unsigned int N;
	unsigned int M;
	unsigned int F;
	unsigned int CD_K;
	double lrate; // learning rate


	record_array *ptr_test_data;
	record_array *ptr_train_data;
	record_array *ptr_qual_data;


	unordered_map<unsigned int, int*> train_map;
	unordered_map<unsigned int, int*> test_map;
	unordered_map<unsigned int, int*> qual_map;

	vector<unsigned int> train_vec;
	vector<unsigned int> test_vec;
	vector<unsigned int> qual_vec;

	RBM() {
		F = 50;
		C = 10;
		M = 17770 / 1 + 1;
		N = 458293 / 1;

		// A = randu<cube>(K, C, M) / 8.0;
		// B = randu<mat>(C, F) / 8.0;
		// BV = randu<mat>(K, M) / 8.0;
		// BH = randu<vec>(F) / 8.0;


		CD_K = 1;
		lrate = 0.05 / BATCH_SIZE;
	}

	void fit(const record_array &train_data, unsigned int n_iter = 1) {

		float* dev_A;
		float* dev_B;
		float* dev_BV;
		float* dev_BH;

		cudaMalloc((void**)&dev_A, sizeof(float) * K * C * M);
		cudaMalloc((void**)&dev_B, sizeof(float) * C * F);
		cudaMalloc((void**)&dev_BV, sizeof(float) * K * M);
		cudaMalloc((void**)&dev_BH, sizeof(float) * F);

		// fill in uniformly distributed random numbers
		GPU_fill_rand(dev_A, K, C*M);
		GPU_fill_rand(dev_B, C, F);
		GPU_fill_rand(dev_BV, K, M);
		GPU_fill_rand(dev_BH, F, 1);

		int* users;
		int* starts;
		int* sizes;


		users = new int[BATCH_SIZE];
		starts = new int[BATCH_SIZE];
		sizes = new int[BATCH_SIZE];


		int* dev_users;
		int* dev_starts;
		int* dev_sizes;

		cudaMalloc((void**)&dev_users, BATCH_SIZE * sizeof(int));
		cudaMalloc((void**)&dev_starts, BATCH_SIZE * sizeof(int));
		cudaMalloc((void**)&dev_sizes, BATCH_SIZE * sizeof(int));


		float* dev_Hzeros;
		float* dev_Hts;

		cudaMalloc((void**)&dev_Hzeros, sizeof(float) * F * BATCH_SIZE);
		cudaMalloc((void**)&dev_Hts, sizeof(float) * F * BATCH_SIZE);


		// TODO: change NUM_MOVIES_PER_BATCH when switching to large data
		// set up movies and ratings related data
		float* dev_Vzeros;
		float* dev_Vts;
		float* dev_W_users;

		cudaMalloc((void**)&dev_Vzeros, sizeof(float) * K * NUM_MOVIES_PER_BATCH);
		cudaMalloc((void**)&dev_Vts, sizeof(float) * K * NUM_MOVIES_PER_BATCH);
		cudaMalloc((void**)&dev_W_users, sizeof(float) * K * F * NUM_MOVIES_PER_BATCH);

		int* movies;
		int* ratings;
		movies = new int[NUM_MOVIES_PER_BATCH];
		ratings = new int[NUM_MOVIES_PER_BATCH];

		int* dev_movies;
		int* dev_ratings;
		cudaMalloc((void**)&dev_movies, NUM_MOVIES_PER_BATCH * sizeof(int));
		cudaMalloc((void**)&dev_ratings, NUM_MOVIES_PER_BATCH * sizeof(int));




		int *ids;
		
		for (unsigned int iter_num = 0; iter_num < n_iter; iter_num++) {
			cout << "working on iteration " << iter_num << "..." << endl;

			// customize CD_K based on the number of iteration
			if (iter_num < 15) CD_K = 1;
			else if (iter_num < 25) CD_K = 3;
			else if (iter_num < 35) CD_K = 5;
			else CD_K = 9;


			// TODO: predict test/qual set & save results






			// train
			int thread_id = 0;
			int size;
			int accu_idx = 0;
			int batch_start = 0;


			for (auto &user_id : train_vec) {
				ids = train_map[user_id];
				size = ids[1] - ids[0];

				users[thread_id] = user_id;
				starts[thread_id] = accu_idx;
				sizes[thread_id] = size;


				
				accu_idx += size;
				thread_id++;

				if (thread_id == BATCH_SIZE) {


					// copy data from host to device
					for (int ib = 0; ib < accu_idx; ib++) {
						record r = train_data.data[batch_start + ib];
						movies[ib] = r.movie;
						ratings[ib] = r.score;

					}

					cudaMemcpy(dev_users, users, sizeof(int) * BATCH_SIZE, cudaMemcpyHostToDevice);
					cudaMemcpy(dev_starts, starts, sizeof(int) * BATCH_SIZE, cudaMemcpyHostToDevice);
					cudaMemcpy(dev_sizes, sizes, sizeof(int) * BATCH_SIZE, cudaMemcpyHostToDevice);

					cudaMemcpy(dev_movies, movies, accu_idx * sizeof(int), cudaMemcpyHostToDevice);
					cudaMemcpy(dev_ratings, ratings, accu_idx * sizeof(int), cudaMemcpyHostToDevice);


					cudaMemset(dev_Vzeros, 0, sizeof(float) * K * accu_idx);
					cudaMemset(dev_Vts, 0, sizeof(float) * K * accu_idx);


					// train batch data
					train(dev_users, dev_movies, dev_ratings, dev_starts, dev_sizes, 
						dev_A, dev_B, dev_BV, dev_BH, F, C,
						dev_Vzeros, dev_Vts, dev_Hzeros, dev_Hts, dev_W_users,
						BATCH_SIZE, CD_K);



					// reset thread_id and accu_idx for next batch
					batch_start += accu_idx;
					thread_id = 0;
					accu_idx = 0;

				}
			}



			// TODO: train last round
			if (thread_id != 0) {
				;
			}


		}

		cout << "finish training" << endl;

		// free memory
		delete[] users;
		delete[] starts;
		delete[] sizes;
		
		cudaFree(dev_starts);
		cudaFree(dev_sizes);
		cudaFree(dev_users);

		cudaFree(dev_A);
		cudaFree(dev_B);
		cudaFree(dev_BV);
		cudaFree(dev_BH);

		cudaFree(dev_Hzeros);
		cudaFree(dev_Hts);


		// free movies and ratings memory
		delete[] movies;
		delete[] ratings;
		cudaFree(dev_movies);
		cudaFree(dev_ratings);
		cudaFree(dev_Vzeros);
		cudaFree(dev_Vts);
		cudaFree(dev_W_users);



	}

};
















/*
NOTE: You can use this macro to easily check cuda error codes
and get more information.

Modified from:
http://stackoverflow.com/questions/14038589/
what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
*/
#define gpuErrChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code,
                      const char *file,
                      int line,
                      bool abort=true) {
  if (code != cudaSuccess) {
    fprintf(stderr,"GPUassert: %s %s %d\n",
            cudaGetErrorString(code), file, line);
    exit(code);
  }
}

// timing setup code
cudaEvent_t start;
cudaEvent_t stop;

#define START_TIMER() {                         \
      gpuErrChk(cudaEventCreate(&start));       \
      gpuErrChk(cudaEventCreate(&stop));        \
      gpuErrChk(cudaEventRecord(start));        \
    }

#define STOP_RECORD_TIMER(name) {                           \
      gpuErrChk(cudaEventRecord(stop));                     \
      gpuErrChk(cudaEventSynchronize(stop));                \
      gpuErrChk(cudaEventElapsedTime(&name, start, stop));  \
      gpuErrChk(cudaEventDestroy(start));                   \
      gpuErrChk(cudaEventDestroy(stop));                    \
  }

////////////////////////////////////////////////////////////////////////////////
// Start non boilerplate code


// void preprocess_data() {
// 	// string train_file_name = "../../../data/mini_main.data";
// 	// string test_file_name = "../../../data/mini_prob.data";
// 	// string qual_file_name = "../../../data/mini_prob.data"; // TODO: Change this name!!!

// 	// string train_file_name = "../../../data/main_data.data";
// 	// string test_file_name = "../../../data/prob_data.data";
// 	// string qual_file_name = "../../../data/qual_data.data";

// 	string train_file_name = "data/mid_main.data";
// 	string test_file_name = "data/mid_prob.data";
// 	string qual_file_name = "data/mid_prob.data";

// 	record_array train_data;
// 	record_array test_data;
// 	record_array qual_data;

// 	train_data.load(train_file_name.c_str());
// 	test_data.load(test_file_name.c_str());
// 	qual_data.load(qual_file_name.c_str());
// 	cout << "finish loading " << qual_file_name << endl;

// }


void rbm_core() {
	;
}

void make_pre_map_vec(const record_array &record_data, unordered_map<unsigned int, int*> &record_map, vector<unsigned int> &record_vec) {
	// unordered_map<unsigned int, int*> record_map;

	unsigned int cur_user = record_data.data[0].user;
	int cur_start = 0;
	int cur_end = 1;
	int* user_ids;
	for (int i = 0; i < record_data.size; i++) {
		record this_data = record_data.data[i];
		if (this_data.user != cur_user) {
			cur_end = i;
			

			user_ids = new int[2];
			user_ids[0] = cur_start;
			user_ids[1] = cur_end;
			record_map[cur_user] = user_ids;
			record_vec.push_back(cur_user);
			
			cur_user = this_data.user;
			cur_start = i;
		}
	}
	if (cur_user != record_vec.back()) {
		user_ids = new int[2];
		user_ids[0] = cur_start;
		user_ids[1] = record_data.size;
		record_map[cur_user] = user_ids;
		record_vec.push_back(cur_user);
	}

	cout << "number of users = " << record_map.size() << endl;

}





int main(int argc, char** argv) {

	unsigned int ITER_NUM = 3;

	string train_file_name = "data/mid_main.data";
	string test_file_name = "data/mid_prob.data";
	string qual_file_name = "data/mid_prob.data";

	// string train_file_name = "data/main_data.data";
	// string test_file_name = "data/prob_data.data";
	// string qual_file_name = "data/qual_data.data";

	record_array train_data;
	record_array test_data;
	record_array qual_data;

	train_data.load(train_file_name.c_str());
	test_data.load(test_file_name.c_str());
	qual_data.load(qual_file_name.c_str());
	cout << "finish loading " << qual_file_name << endl;


	RBM rbm;
	rbm.ptr_train_data = &train_data;
	rbm.ptr_test_data = &test_data;
	rbm.ptr_qual_data = &qual_data;

	make_pre_map_vec(train_data, rbm.train_map, rbm.train_vec);
	make_pre_map_vec(test_data, rbm.test_map, rbm.test_vec);
	make_pre_map_vec(qual_data, rbm.qual_map, rbm.qual_vec);

	rbm.fit(train_data, ITER_NUM);



	cudaTest();


	if (argc == 2) {
	ifstream ifs(argv[1]);
	// stringstream buffer;
	// buffer << ifs.rdbuf();
	// cluster(buffer, k, batch_size);
	}
}




