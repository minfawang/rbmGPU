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

// #include <random>
// #include <unordered_map>
// #include <vector>
#include <set>

#include <cuda_runtime.h>
#include <cuda.h>
#include <curand.h>
#include "rbm_cuda.cuh"



#define NUM_THREADS 8
#define BATCH_SIZE (NUM_THREADS * 1024)

#define maxBlocks 200
#define threadsPerBlock 512

// // TODO: change NUM_MOVIES_PER_BATCH when switching to large data
// // number of movies per batch in mid_data: max = 97527, min = 86203
// // in full data: max = 493495, min = 402065
// #define NUM_MOVIES_PER_BATCH 97527
// // #define NUM_MOVIES_PER_BATCH 493495



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

// float RMSE(const record_array & test_data, const float* & prediction) {
// 	double s = 0;
// 	for (int i = 0; i < test_data.size; i++) {
// 		s += (test_data[i].score - prediction[i]) * (test_data[i].score - prediction[i]);
// 	}
// 	return sqrt(s / test_data.size);
// }




class RBM {
public:
	unsigned int C;
	unsigned int N;
	unsigned int M;
	// unsigned int F;
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

	float* dev_results; // num_records

	int* train_movies;
	int* train_ratings;

	unsigned int MAX_NUM_MOVIES_IN_BATCH;
	curandGenerator_t gen;

	RBM(string train_file_name, string test_file_name, string qual_file_name) {

		record_array train_data;
		record_array test_data;
		record_array qual_data;

		train_data.load(train_file_name.c_str());
		test_data.load(test_file_name.c_str());
		qual_data.load(qual_file_name.c_str());
		cout << "finish loading " << qual_file_name << endl;


		train_movies = new int[train_data.size];
		train_ratings = new int[train_data.size];

		setNumUserMovieOthers(train_data, &N, &M, &MAX_NUM_MOVIES_IN_BATCH, train_movies, train_ratings);
		cout << "train: " << train_data.size << "\ttest: " << test_data.size << "\tqual: " << qual_data.size << endl;
		cout << "users: " << N << "\tmovies: " << M << "\tMAX_NUM_MOVIES_IN_BATCH: " << MAX_NUM_MOVIES_IN_BATCH << endl;

		C = 10;
		CD_K = 1;

		lrate = 0.01 * M / train_data.size;
		lrate_BH = lrate / BATCH_SIZE;


		train_vec = make_pre_vec(train_data, N);
		test_vec = make_pre_vec(test_data, N);
		qual_vec = make_pre_vec(qual_data, N);

		cout << "finish making pre vecs" << endl;


		// Create a pseudo-random number generator
		const float mean = 0.0;
		const float std_dev = 0.1;
		curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
		curandSetPseudoRandomGeneratorSeed(gen, (unsigned long long) clock());

		cudaMalloc(&dev_W, sizeof(float) * K * F * M);
		cudaMalloc(&dev_BV, sizeof(float) * K * M);
		cudaMalloc(&dev_BH, sizeof(float) * F);

		curandGenerateNormal(gen, dev_W, K * F * M, mean, std_dev);
		curandGenerateNormal(gen, dev_BH, F, mean, std_dev);
		curandGenerateNormal(gen, dev_BV, K * M, mean, std_dev);



		cudaMalloc(&dev_W_inc, sizeof(float) * K * F * M);
		cudaMalloc(&dev_BV_inc, sizeof(float) * K * M);
		cudaMalloc(&dev_BH_inc, sizeof(float) * F * BATCH_SIZE);



		// TEST CODE
		float* BV = new float[K * M];
		cudaMemcpy(BV, dev_BV, sizeof(float) * K * M, cudaMemcpyDeviceToHost);
		for (unsigned int i = 150; i < 160; i++) {
			cout << BV[i] << " ";
		}
		cout << endl;
		delete[] BV;
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

		
		// TODO: uncomment below
		// cudaFree(dev_results);

		delete[] train_movies;
		delete[] train_ratings;

		curandDestroyGenerator(gen);

	}

	void fit(unsigned int n_iter = 1) {


		// TEST CODE Memory Allocation Area
		float* BV_inc = new float[K * M];
		float* BH_inc = new float[F * BATCH_SIZE];
		float* V = new float[K * MAX_NUM_MOVIES_IN_BATCH];
		float* H = new float[F * BATCH_SIZE];



		// allocate memory to store movies and ratings
		int* dev_train_vec_in_batch;
		int* dev_movies_in_batch;
		int* dev_ratings_in_batch;

		cudaMalloc(&dev_train_vec_in_batch, sizeof(int) * 2 * BATCH_SIZE);
		cudaMalloc(&dev_movies_in_batch, sizeof(int) * MAX_NUM_MOVIES_IN_BATCH);
		cudaMalloc(&dev_ratings_in_batch, sizeof(int) * MAX_NUM_MOVIES_IN_BATCH);


		// allocate memory for V0, Vt, H0, Ht ...
		float* dev_Vzeros;
		float* dev_Vts;
		float* dev_Hzeros;
		float* dev_Hts;

		cudaMalloc(&dev_Vzeros, sizeof(float) * K * MAX_NUM_MOVIES_IN_BATCH);
		cudaMalloc(&dev_Vts, sizeof(float) * K * MAX_NUM_MOVIES_IN_BATCH);
		cudaMalloc(&dev_Hzeros, sizeof(float) * F * BATCH_SIZE);
		cudaMalloc(&dev_Hts, sizeof(float) * F * BATCH_SIZE);



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
				dev_Vzeros, dev_Vts, dev_Hzeros, dev_Hts,
				dev_W, dev_BV, dev_BH, dev_W_inc, dev_BV_inc, dev_BH_inc,
				batch_size, num_movies_in_this_batch, i_batch_start, 
				M, lrate, lrate_BH,
				blocks, threadsPerBlock);




			// TEST CODE: verify BV_inc
			// cout << "movies in batch = " << num_movies_in_this_batch << endl;

			// cudaMemcpy(H, dev_Hzeros, sizeof(float) * F * batch_size, cudaMemcpyDeviceToHost);
			// for (unsigned int i = F * 100; i < F * 101; i++) {
			// 	cout << H[i] << " ";
			// }
			// cout << endl;
			// cudaMemcpy(V, dev_Vts, sizeof(float) * K * num_movies_in_this_batch, cudaMemcpyDeviceToHost);
			// for (unsigned int i = K * 100; i < K * 110; i++) {
			// 	cout << V[i] << " ";
			// }
			// cout << endl;
			// cudaMemcpy(BH_inc, dev_BH_inc, sizeof(float) * F * batch_size, cudaMemcpyDeviceToHost);
			// for (unsigned int i = F * 100; i < F * 101; i++) {
			// 	cout << BH_inc[i] << " ";
			// }
			// cout << endl;
			cudaMemcpy(BV_inc, dev_BV_inc, sizeof(float) * K * M, cudaMemcpyDeviceToHost);
			for (unsigned int i = K * 110; i < K * 112; i++) {
				cout << BV_inc[i] << " ";
			}
			cout << endl;


			// TODO: implement the kernel code to update W, BV, BH
		}


		// TEST CODE memory release area
		delete[] BV_inc;
		delete[] BH_inc;
		delete[] V;
		delete[] H;


		cudaFree(dev_train_vec_in_batch);
		cudaFree(dev_movies_in_batch);
		cudaFree(dev_ratings_in_batch);


		cudaFree(dev_Vzeros);
		cudaFree(dev_Vts);
		cudaFree(dev_Hzeros);
		cudaFree(dev_Hts);




		// // generate copy of B in order to do update
		// float* dev_B_old;
		// cudaMalloc((void**)&dev_B_old, sizeof(float) * K * F);
		// cudaMemcpy(dev_B_old, dev_B, sizeof(float) * K * F, cudaMemcpyDeviceToDevice);

		// int* users;
		// int* starts;
		// int* sizes;


		// users = new int[BATCH_SIZE];
		// starts = new int[BATCH_SIZE];
		// sizes = new int[BATCH_SIZE];


		// int* dev_users;
		// int* dev_starts;
		// int* dev_sizes;

		// cudaMalloc((void**)&dev_users, BATCH_SIZE * sizeof(int));
		// cudaMalloc((void**)&dev_starts, BATCH_SIZE * sizeof(int));
		// cudaMalloc((void**)&dev_sizes, BATCH_SIZE * sizeof(int));


		// float* dev_Hzeros;
		// float* dev_Hts;

		// cudaMalloc((void**)&dev_Hzeros, sizeof(float) * F * BATCH_SIZE);
		// cudaMalloc((void**)&dev_Hts, sizeof(float) * F * BATCH_SIZE);


		// // TODO: change NUM_MOVIES_PER_BATCH when switching to large data
		// // set up movies and ratings related data
		// float* dev_Vzeros;
		// float* dev_Vts;
		// float* dev_W_users;

		// cudaMalloc((void**)&dev_Vzeros, sizeof(float) * K * NUM_MOVIES_PER_BATCH);
		// cudaMalloc((void**)&dev_Vts, sizeof(float) * K * NUM_MOVIES_PER_BATCH);
		// cudaMalloc((void**)&dev_W_users, sizeof(float) * K * F * NUM_MOVIES_PER_BATCH);

		// int* movies;
		// int* ratings;
		// movies = new int[NUM_MOVIES_PER_BATCH];
		// ratings = new int[NUM_MOVIES_PER_BATCH];

		// int* dev_movies;
		// int* dev_ratings;
		// cudaMalloc((void**)&dev_movies, NUM_MOVIES_PER_BATCH * sizeof(int));
		// cudaMalloc((void**)&dev_ratings, NUM_MOVIES_PER_BATCH * sizeof(int));




		// int *ids;
		
		// for (unsigned int iter_num = 0; iter_num < n_iter; iter_num++) {
		// 	cout << "working on iteration " << iter_num << "..." << endl;

		// 	// customize CD_K based on the number of iteration
		// 	if (iter_num < 15) CD_K = 1;
		// 	else if (iter_num < 25) CD_K = 3;
		// 	else if (iter_num < 35) CD_K = 5;
		// 	else CD_K = 9;


		// 	// TODO: predict test/qual set & save results






		// 	// train
		// 	int thread_id = 0;
		// 	int size;
		// 	int accu_idx = 0;
		// 	int batch_start = 0;


		// 	// for (auto &user_id : train_vec) {
		// 	for (int iv = 0; iv < train_vec.size(); iv++) {
		// 		unsigned int user_id = train_vec[iv];

		// 		ids = train_map[user_id];
		// 		size = ids[1] - ids[0];

		// 		users[thread_id] = user_id;
		// 		starts[thread_id] = accu_idx;
		// 		sizes[thread_id] = size;


				
		// 		accu_idx += size;
		// 		thread_id++;

		// 		if (thread_id == BATCH_SIZE || iv == (train_vec.size() - 1)) {


		// 			// copy data from host to device
		// 			for (int ib = 0; ib < accu_idx; ib++) {
		// 				record r = train_data.data[batch_start + ib];
		// 				movies[ib] = r.movie;
		// 				ratings[ib] = r.score;

		// 			}

		// 			cudaMemcpy(dev_users, users, sizeof(int) * thread_id, cudaMemcpyHostToDevice);
		// 			cudaMemcpy(dev_starts, starts, sizeof(int) * thread_id, cudaMemcpyHostToDevice);
		// 			cudaMemcpy(dev_sizes, sizes, sizeof(int) * thread_id, cudaMemcpyHostToDevice);

		// 			cudaMemcpy(dev_movies, movies, accu_idx * sizeof(int), cudaMemcpyHostToDevice);
		// 			cudaMemcpy(dev_ratings, ratings, accu_idx * sizeof(int), cudaMemcpyHostToDevice);


		// 			cudaMemset(dev_Vzeros, 0, sizeof(float) * K * accu_idx);
		// 			cudaMemset(dev_Vts, 0, sizeof(float) * K * accu_idx);


		// 			// train batch data
		// 			train(dev_users, dev_movies, dev_ratings, dev_starts, dev_sizes, 
		// 				dev_A, dev_B, dev_B_old, dev_BV, dev_BH, F, C, &lrate,
		// 				dev_Vzeros, dev_Vts, dev_Hzeros, dev_Hts, dev_W_users,
		// 				thread_id, CD_K);



		// 			// TEST CODE
		// 			// check B
		// 			cudaMemcpy(B, dev_B, sizeof(float) * C * F, cudaMemcpyDeviceToHost);
		// 			for (int f = 0; f < F; f++) {
		// 				for (int c = 0; c < C; c++) {
		// 					cout << B[f * C + c] << " ";
		// 				}
		// 				cout << endl;
		// 			}



		// 			// reset thread_id and accu_idx for next batch
		// 			batch_start += accu_idx;
		// 			thread_id = 0;
		// 			accu_idx = 0;

		// 		}
		// 	}


		// }

		// cout << "finish training" << endl;


		// // // TEST CODE
		// // cudaMemcpy(B, dev_B, sizeof(float) * C * F, cudaMemcpyDeviceToHost);
		// // ofstream out_file("test_coeff.txt");
		// // for (int f = 0; f < F; f++) {
		// // 	for (int c = 0; c < C; c++) {
		// // 		out_file << B[f * C + c] << " ";
		// // 	}
		// // 	out_file << endl;
		// // }
		// // out_file.close();

		// // free memory
		// delete[] users;
		// delete[] starts;
		// delete[] sizes;
		
		// cudaFree(dev_starts);
		// cudaFree(dev_sizes);
		// cudaFree(dev_users);


		// cudaFree(dev_Hzeros);
		// cudaFree(dev_Hts);


		// // free movies and ratings memory
		// delete[] movies;
		// delete[] ratings;
		// cudaFree(dev_movies);
		// cudaFree(dev_ratings);
		// cudaFree(dev_Vzeros);
		// cudaFree(dev_Vts);
		// cudaFree(dev_W_users);



	}





	void setNumUserMovieOthers(const record_array &rcd_array, unsigned int* ptr_N, unsigned int* ptr_M, unsigned int* ptr_MAX_NUM_MOVIES_IN_BATCH, int* movies, int* ratings) {
		set<int> movies_set;
		set<int> users_set;

		unsigned int cur_user = rcd_array.data[0].user;
		int num_users = 0;
		int num_movies_in_this_batch = 0;

		int batch_num = 1;
		*ptr_MAX_NUM_MOVIES_IN_BATCH = 0;

		for (int i = 0; i < rcd_array.size; i++) {
			record r = rcd_array.data[i];
			if (movies_set.find(r.movie) == movies_set.end()) {
				movies_set.insert(r.movie);
			}

			if (users_set.find(r.user) == users_set.end()) {
				users_set.insert(r.user);
			}

			if (r.user != cur_user) {
				cur_user = r.user;
				num_users++;
				// if (num_users % BATCH_SIZE == 0) {
				// 	*ptr_MAX_NUM_MOVIES_IN_BATCH = max(*ptr_MAX_NUM_MOVIES_IN_BATCH, (unsigned int)num_movies_in_this_batch);
				// 	num_movies_in_this_batch = 0;
				// }
			}

			train_movies[i] = r.movie - 1;
			train_ratings[i] = r.score;

			if ((int)cur_user > batch_num * BATCH_SIZE) {
				*ptr_MAX_NUM_MOVIES_IN_BATCH = max(*ptr_MAX_NUM_MOVIES_IN_BATCH, (unsigned int)num_movies_in_this_batch);
				num_movies_in_this_batch = 0;

				batch_num++;
			}
			num_movies_in_this_batch++;
		}

		*ptr_N = users_set.size();
		*ptr_M = movies_set.size();
		*ptr_MAX_NUM_MOVIES_IN_BATCH = max(*ptr_MAX_NUM_MOVIES_IN_BATCH, (unsigned int)num_movies_in_this_batch);
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
				user_id = rcd_array.data[end].user - 1;
			}

			rcd_vec[2 * vec_idx + 1] = end;

			start = end;
		}

		return rcd_vec;
	}
};


























int main(int argc, char** argv) {

	unsigned int ITER_NUM = 3;

	string train_file_name = "data/mid_main.data";
	string test_file_name = "data/mid_prob.data";
	string qual_file_name = "data/mid_prob.data";

	// string train_file_name = "data/main_data.data";
	// string test_file_name = "data/prob_data.data";
	// string qual_file_name = "data/qual_data.data";


	RBM rbm(train_file_name, test_file_name, qual_file_name);

	rbm.fit(ITER_NUM);




	if (argc == 2) {
	ifstream ifs(argv[1]);
	// stringstream buffer;
	// buffer << ifs.rdbuf();
	// cluster(buffer, k, batch_size);
	}
}




