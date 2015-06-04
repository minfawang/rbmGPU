#include <cstring>


#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sys/stat.h>
#include <random>
#include <cmath>
#include <set>

#include <omp.h>
#include <ctime>

// #include <armadillo>
// #include <shark/Unsupervised/RBM/BinaryRBM.h>


#ifndef __RBM_ESTIMATORS
#define __RBM_ESTIMATORS



#define NUM_THREADS 8
#define BATCH_SIZE (NUM_THREADS * 256)
#define K 5


using namespace std;
// using namespace arma;


// bool isFuckedUp(double num) {
// 	return isnan(num) || isinf(num);
// }


double sigma(double num) {
	return 1.0 / (1 + exp(-num));
}

void fillRandomArray(float* a, int size) {
	// the numbers will be uniformly distributed between -0.1 and 0.1
#pragma omp parallel for num_threads(NUM_THREADS)
	for (int i = 0 ; i < size; i++) {
		a[i] = (double) rand() / RAND_MAX / 5 - 0.1;
	}
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
	unsigned int C;
	unsigned int N;
	unsigned int M;
	unsigned int F;
	unsigned int CD_K;

	float lrate;
	float lrate_BH;

	record_array *ptr_test_data;
	record_array *ptr_train_data;
	record_array *ptr_qual_data;


	vector<int*> train_vec;
	vector<int*> test_vec;
	vector<int*> qual_vec;


	float* W;
	float* BV;
	float* BH;

	int MAX_NUM_MOVIES_IN_BATCH;
	float* W_inc;
	float* BV_inc;
	float* BH_inc;

	vector<float> results;


	RBM(record_array &train_data, record_array &test_data, record_array &qual_data) {

		ptr_train_data = &train_data;
		ptr_test_data = &test_data;
		ptr_qual_data = &qual_data;

		setNumUserMovieOthers(train_data, &N, &M, &MAX_NUM_MOVIES_IN_BATCH);
		cout << "train: " << train_data.size << "\ttest: " << test_data.size << "\tqual: " << qual_data.size << endl;
		cout << "users: " << N << "\tmovies: " << M << "\tMAX_NUM_MOVIES_IN_BATCH: " << MAX_NUM_MOVIES_IN_BATCH << endl;
		F = 100;
		C = 10;
		CD_K = 1;

		lrate = 0.01 * M / train_data.size;
		lrate_BH = lrate / BATCH_SIZE;



		train_vec = make_pre_vec(train_data, N);
		test_vec = make_pre_vec(test_data, N);
		qual_vec = make_pre_vec(qual_data, N);


		W = new float[K * F * M];
		BV = new float[K * M];
		BH = new float[F];

		fillRandomArray(W, K * F * M);
		fillRandomArray(BV, K * M);
		fillRandomArray(BH, F);

		// W_inc = new float[K * F * MAX_NUM_MOVIES_IN_BATCH];
		// BV_inc = new float[K * MAX_NUM_MOVIES_IN_BATCH];
		W_inc = new float[K * F * M];
		BV_inc = new float[K * M];
		BH_inc = new float[F * BATCH_SIZE];

		cout << "finish pre-processing" << endl;
	}

	~RBM() {
		delete[] W;
		delete[] BV;
		delete[] BH;

		delete[] W_inc;
		delete[] BV_inc;
		delete[] BH_inc;
	}

	void fit(const record_array &train_array, unsigned int n_iter = 1) {

		float train_time = 0;
		float predict_time = 0;
		float milliseconds;
		time_t timer_start, timer_stop;

		for (int iter_num = 0; iter_num < n_iter; iter_num++) {

			// start timer to record predicting process
			timer_start = clock();

			// predict list
			predict_list(*ptr_test_data, test_vec, train_array);
			timer_stop = clock();
			milliseconds = 1000 * (timer_stop - timer_start) / (float) CLOCKS_PER_SEC;
			predict_time += milliseconds;
			cout << "time spent on predicting = " << milliseconds << "\t" << flush;


			cout << "RMSE: " << RMSE(*ptr_test_data, results) << endl;

			// start timer to record training process
			timer_start = clock();

			cout << "iteration " << iter_num << "\t" << flush;

			for (int iv = 0; iv < N; iv += BATCH_SIZE) {
				if (iv % 3 == 0)
					cout << "." << flush;

				// determinine batch size
				int real_BATCH_SIZE = min((unsigned int)BATCH_SIZE, N - iv);

				// // set up weights increments to zero
				// int batch_movie_start = train_vec[iv][0];
				// memset(W_inc, 0, sizeof(float) * K * F * MAX_NUM_MOVIES_IN_BATCH);
				// memset(BV_inc, 0, sizeof(float) * K * MAX_NUM_MOVIES_IN_BATCH);
				memset(W_inc, 0, sizeof(float) * K * F * M);
				memset(W_inc, 0, sizeof(float) * K * M);
				memset(BH_inc, 0, sizeof(float) * F * BATCH_SIZE);


				for (int thread_idx = 0; thread_idx < real_BATCH_SIZE; thread_idx++) {

					int user_id = iv + thread_idx;
					int* movies_segment = train_vec[user_id];
					int start = movies_segment[0];
					int size = movies_segment[1];


					if (size != 0) {
						train(train_array.data + start, user_id, size, thread_idx);
						// cout << "train on " << user_id << " of size " << size << endl;
					}
				}

				// update weights after the mini-batch
				// update W
				for (int i = 0; i < M; i++) {
					for (int j = 0; j < F; j++) {
						for (int k = 0; k < K; k++) {
							W[i * F * K + j + K + k] += lrate * W_inc[i * F * K + j + K + k];
						}
					}
				}

				// update BV
				for (int i = 0; i < M; i++) {
					for (int k = 0; k < K; k++) {
						BV[i * K + k] += lrate * BV_inc[i * K + k];
					}
				}

				// update BH
				for (int thread_idx = 0; thread_idx < real_BATCH_SIZE; thread_idx++) {
					for (int j = 0; j < F; j++) {
						BH[j] += lrate_BH * BH_inc[thread_idx * F + j];
					}
				}

			}

			// stop training timer
			timer_stop = clock();
			milliseconds = 1000 * (timer_stop - timer_start) / (float)CLOCKS_PER_SEC;
			train_time += milliseconds;
			cout << "time spent on training = " << milliseconds << endl;
		}


		// compute average time
		cout << "Overall speed info per iteration (ms): " << endl;
		cout << "averaged predicting time = " << predict_time / n_iter << endl;
		cout << "averaged training time = " << train_time / n_iter << endl;

	}

	void train(const record* data, int user_id, int size, int thread_idx) {
		// set up Gibbs sampling steps
		// int T = 1;
		// if (updateWeights) { T = CD_K;}

		// initialize vectors to train
		float V0[K * size];
		float Vt[K * size];
		float H0[F];
		float Ht[F];

		int ims[size];

		// set up V0 and ims
		memset(V0, 0, sizeof(float) * K * size);
		for (int i = 0; i < size; i++) {
			record r = data[i];
			V0[i * K + int(r.score) - 1] = 1;
			ims[i] = r.movie - 1;
		}

		//////////////// positive phase ////////////////
		// compute H0
		memcpy(H0, BH, sizeof(float) * F);
		// memcpy(Ht, BH, sizeof(float) * F);
		for (int i = 0; i < size; i++) {
			float* W_movie = W + ims[i] * (K * F);

			for (int j = 0; j < F; j++) {
				for (int k = 0; k < K; k++) {
					H0[j] += W_movie[j * K + k] * V0[i * K + k];
				}
			}
		}

		// logistic function on H
		for (int j = 0; j < F; j++) {
			H0[j] = 1.0 / (1 + exp(-H0[j]));
		}

		//////////////// negative phase ////////////////
		// memcpy(Vt, BV, sizeof(float) * K * size);
		for (int i = 0; i < size; i++) {
			float* W_movie = W + ims[i] * (K * F);
			memcpy(Vt + i * K, BV + ims[i] * K, sizeof(float) * K);

			for (int j = 0; j < F; j++) {
				for (int k = 0; k < K; k++) {
					Vt[i * K + k] += H0[j] * W_movie[j * K + k];
				}
			}

			// normalize Vt
			float sum_k = 0.0;
			for (int k = 0; k < K; k++) {
				Vt[i * K + k] = exp(Vt[i * K + k]);
				sum_k += Vt[i * K + k];
			}
			for (int k = 0; k < K; k++) {
				Vt[i * K + k] /= sum_k;
			}
		}


			
		// compute Ht
		memcpy(Ht, BH, sizeof(float) * F);
		for (int i = 0; i < size; i++) {
			float* W_movie = W + ims[i] * (K * F);

			for (int j = 0; j < F; j++) {
				for (int k = 0; k < K; k++) {
					Ht[j] += W_movie[j * K + k] * Vt[i * K + k];
				}
			}
		}

		// logistic function on H
		for (int j = 0; j < F; j++) {
			Ht[j] = 1.0 / (1 + exp(-Ht[j]));
		}


		//////////////// TODO: update weight increments ////////////////

		// update BV_inc
		for (int i = 0; i < size; i++) {
			for (int k = 0; k < K; k++) {
				BV_inc[ims[i] * K + k] += (V0[i * K + k] - Vt[i * K + k]);
			}
		}

		// update W_inc
		for (int i = 0; i < size; i++) {
			for (int j = 0; j < F; j++) {
				for (int k = 0; k < K; k++) {
					W_inc[ims[i] * K * F + j * K + k] += (H0[j] * V0[i * K + k] - Ht[j] * Vt[i * K + k]);
				}
			}
		}

		// update BH_inc
		for (int j = 0; j < F; j++) {
			BH_inc[thread_idx * F + j] = (H0[j] - Ht[j]);
		}

	}





	void predict_list(const record_array &predict_array, const vector<int*> predict_vec, const record_array &train_array) {
		if (results.size() != predict_array.size) {
			results.resize(predict_array.size);
		}

		for (int iv = 0; iv < N; iv += BATCH_SIZE) {
			// determinine batch size
			int real_BATCH_SIZE = min((unsigned int)BATCH_SIZE, N - iv);

			for (int thread_idx = 0; thread_idx < real_BATCH_SIZE; thread_idx++) {

				int user_id = iv + thread_idx;


				int* predict_movies_segment = predict_vec[user_id];
				int predict_start = predict_movies_segment[0];
				int predict_size = predict_movies_segment[1];

				if (predict_size != 0) {

					int* movies_segment = train_vec[user_id];
					int start = movies_segment[0];
					int size = movies_segment[1];
					
					predict_user(user_id, train_array.data + start, size, predict_array.data + predict_start, predict_start, predict_size);
					// cout << "train on " << user_id << " of size " << size << endl;
				}
			}
		}
	}

	void predict_user(int user_id, const record* data, int size, const record* predict_data, int predict_start, int predict_size) {
		// set up Gibbs sampling steps
		// int T = 1;
		// if (updateWeights) { T = CD_K;}

		// initialize vectors to train
		float V0[K * size];
		float Vt[K * predict_size];
		float H0[F];

		int ims[size];
		int predict_im;


		// set up V0 and ims
		memset(V0, 0, sizeof(float) * K * size);
		for (int i = 0; i < size; i++) {
			record r = data[i];
			V0[i * K + int(r.score) - 1] = 1;
			ims[i] = r.movie - 1;
		}

		//////////////// positive phase ////////////////
		// compute H0
		memcpy(H0, BH, sizeof(float) * F);
		// memcpy(Ht, BH, sizeof(float) * F);
		for (int i = 0; i < size; i++) {
			float* W_movie = W + ims[i] * (K * F);

			for (int j = 0; j < F; j++) {
				for (int k = 0; k < K; k++) {
					H0[j] += W_movie[j * K + k] * V0[i * K + k];
				}
			}
		}

		// logistic function on H
		for (int j = 0; j < F; j++) {
			H0[j] = 1.0 / (1 + exp(-H0[j]));
		}

		//////////////// negative phase ////////////////
		for (int i = 0; i < predict_size; i++) {

			predict_im = predict_data[i].movie - 1;

			float* W_movie = W + predict_im * (K * F);
			memcpy(Vt + i * K, BV + predict_im * K, sizeof(float) * K);

			for (int j = 0; j < F; j++) {
				for (int k = 0; k < K; k++) {
					Vt[i * K + k] += H0[j] * W_movie[j * K + k];
				}
			}

			// normalize Vt
			float sum_k = 0.0;
			for (int k = 0; k < K; k++) {
				Vt[i * K + k] = exp(Vt[i * K + k]);
				sum_k += Vt[i * K + k];
			}
			for (int k = 0; k < K; k++) {
				Vt[i * K + k] /= sum_k;
			}


			// compute averaged score
			float avg_score = 0.0;
			for (int k = 0; k < K; k++) {
				avg_score += (k + 1) * Vt[i * K + k];
			}
			results[predict_start + i] = avg_score;
		}

	}




	void setNumUserMovieOthers(const record_array &rcd_array, unsigned int* ptr_N, unsigned int* ptr_M, int* ptr_MAX_NUM_MOVIES_IN_BATCH) {
		set<int> movies_set;
		set<int> users_set;

		*ptr_MAX_NUM_MOVIES_IN_BATCH = 0;
		int cur_user = rcd_array.data[0].user;
		int num_users = 0;
		int num_movies_in_this_batch = 0;

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
				if (num_users % BATCH_SIZE == 0) {
					*ptr_MAX_NUM_MOVIES_IN_BATCH = max(*ptr_MAX_NUM_MOVIES_IN_BATCH, num_movies_in_this_batch);
					num_movies_in_this_batch = 0;
				}
			}
			num_movies_in_this_batch++;
		}

		*ptr_MAX_NUM_MOVIES_IN_BATCH = max(*ptr_MAX_NUM_MOVIES_IN_BATCH, num_movies_in_this_batch);
		*ptr_N = users_set.size();
		*ptr_M = movies_set.size();
	}


	vector<int*> make_pre_vec(const record_array &rcd_array, int N) {
		vector<int*> rcd_vec(N);
		int user_id = rcd_array.data[0].user - 1;
		int array_idx = 0;
		int start = 0;
		int end = 0;

		for (int vec_idx = 0; vec_idx < N; vec_idx++) {
			int* user_info = new int[2];
			user_info[0] = start;

			while ((user_id == vec_idx) && (end < rcd_array.size)) {
				end++;
				user_id = rcd_array.data[end].user - 1;
			}

			user_info[1] = end - start;

			rcd_vec[vec_idx] = user_info;
			start = end;
		}

		return rcd_vec;
	}

};


















int main () {
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

	cout << "finish loading " << train_file_name << endl;




	RBM rbm(train_data, test_data, qual_data);


	unsigned int NUM_ITERS = 5;
	rbm.fit(train_data, NUM_ITERS);





	// vector<float>results = rbm.predict_list(test_data);
	// cout << "RMSE: " << RMSE(test_data, results) << endl;

	// // store results
	// ofstream rbm_out_file;
	// rbm_out_file.open("test_rbm_out.txt");
	// for (int i = 0; i < test_data.size; i++) {
	// 	rbm_out_file << results[i] << endl;
	// }
	// rbm_out_file.close();

}





#endif