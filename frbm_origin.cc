#include <armadillo>
#include <iostream>
#include <omp.h>
#include <unordered_map>

#include "../types.hpp"


#ifndef __RBM_ESTIMATORS
#define __RBM_ESTIMATORS


#define NUM_THREADS 8
#define BATCH_SIZE (NUM_THREADS * 20)



using namespace arma;


bool isFuckedUp(double num) {
	return isnan(num) || isinf(num);
}


double sigma(double num) {
	return 1.0 / (1 + exp(-num));
}

bool file_exists(const char *fileName) {
	struct stat fileInfo;
	return stat(fileName, &fileInfo) == 0;
}

class basic_rbm : public estimator_base {
public:

	cube A;
	mat B;
	mat BV; // K * M
	vec BH; // F
	// mat BH; // K * F

	unsigned int C;
	unsigned int N;
	unsigned int M;
	unsigned int K;
	unsigned int F;
	unsigned int CD_K;
	double lrate; // learning rate


	record_array *ptr_test_data;
	record_array *ptr_train_data;
	record_array *ptr_qual_data;


	unordered_map<unsigned int, int*> train_map;
	unordered_map<unsigned int, int*> test_map;
	unordered_map<unsigned int, int*> qual_map;



	basic_rbm() {
		K = 5;
		F = 50;
		C = 10;
		M = 17770 / 1 + 1; // TODO: change M to be total number of movies
		N = 458293 / 1;

		A = randu<cube>(K, C, M) / 8.0;
		B = randu<mat>(C, F) / 8.0;
		BV = randu<mat>(K, M) / 8.0;
		BH = randu<vec>(F) / 8.0;


		CD_K = 1;
		lrate = 0.04 / BATCH_SIZE;


	}

	virtual bool save(const char * file_name) {
		return true;
	}

	virtual bool load(const char * file_name) {
		return true;
	}


	virtual void fit(const record_array & train_data, unsigned int n_iter = 1, bool countinue_fit = false) {


		// training stage
		for (int iter_num = 0; iter_num < n_iter; iter_num++) {
			// customize CD_K based on the number of iteration
			if (iter_num < 15)
				CD_K = 1;
			else if (iter_num < 25)
				CD_K = 3;
			else if (iter_num < 35)
				CD_K = 5;
			else
				CD_K = 9;
			


			// TEST CODE

			cout << "predicting ... " << endl;
			vector<float> results = predict_array(*ptr_test_data, *ptr_qual_data, test_map, qual_map);
			float prob_rmse = RMSE(*ptr_test_data, results);
			cout << "RMSE: " << prob_rmse << endl;
			// if (prob_rmse < 0.93) {
			// 	predict_qual_results_to_file(*ptr_qual_data, prob_rmse, iter_num);
			// }




			
			cout << "working on iteration " << iter_num << "..." << endl;

			unsigned int user_id = train_data.data[0].user;
			unsigned int start = 0;
			unsigned int end = 0;



			int starts[BATCH_SIZE];
			int ends[BATCH_SIZE];
			int users[BATCH_SIZE];
			int thread_id = 0;
			starts[0] = 0;

			for (int i = 0; i < train_data.size; i++) {
				record r = train_data.data[i];
				if ((user_id != r.user) || i == train_data.size-1) {
					ends[thread_id] = (i == train_data.size-1) ? (i + 1) : i;
					users[thread_id] = user_id;

					user_id = r.user;
					thread_id++;

					// process a batch
					if (thread_id == (BATCH_SIZE) || i == train_data.size-1) {
#pragma omp parallel for num_threads(NUM_THREADS)
						for (int t = 0; t < thread_id; t++) {
							train(train_data.data+starts[t], users[t], ends[t]-starts[t], CD_K);
						}


						thread_id = 0;
					}
					starts[thread_id] = i;
				}
			}


			// // store predicted data to file
			// ofstream out_file;
		 //    out_file.open("test_coeff.txt");

		 //    // store W to file
		 //    for (int i = 0; i < M; i++) {
		 //    	for (int j = 0; j < F; j++) {
		 //    		out_file << W(0, j, i) << " ";
		 //    	}
		 //    	out_file << endl;
		 //    }

		 //    // store BV to file
		 //    for (int i = 0; i < M; i++) {
		 //    	for (int k = 0; k < K; k++) {
		 //    		out_file << BV(k, i) << " ";
		 //    	}
		 //    	out_file << endl;
		 //    }
		    
		 //    // store BH to file
		 //    for (int j = 0; j < F; j++) {
		 //    	out_file << BH(j) << endl;
		 //    }
		    
		 //    out_file.close();
		}


		cout << "finish training!" << endl;
		cout << "train data size: " << ptr_train_data->size << endl;
		cout << "test data size: " << ptr_test_data->size << endl;


	}


	void train(const record *data, unsigned int user_id, unsigned int size, int CD_K) {
		// initialization
		mat V0 = zeros<mat>(K, size);
		mat Vt = zeros<mat>(K, size);
		vec H0 = zeros<vec>(F);
		vec Ht = zeros<vec>(F);

		vector<int> ims(size);
		cube W_user(K, F, size);



		// set up V0 and Vt based on the input data.
		for (int i = 0; i < size; i++) {
			record r = data[i];
			V0(int(r.score)-1, i) = 1; // score - 1 is the index
			Vt(int(r.score)-1, i) = 1;

			ims[i] = r.movie;
			W_user.slice(i) = A.slice(r.movie) * B;
		}

		/*
		/////////////////// set up H0 by V -> H //////////////////
		H0(j) = sigma( BH(j) + sum_ik ( W(k, j, r.movie) * V0(k, i) ))
		*/

		H0 = BH;
		for (int i = 0; i < size; i++) {
			H0 += W_user.slice(i).t() * V0.col(i);
		}
		H0 = 1.0 / (1 + exp(-H0));
		


		/////////////////// Do the contrastive divergence ///////////////////
		for (int n = 0; n < CD_K; n++) {

			////////////// positive phase: V -> H /////////
			Ht = BH;
			for (int i = 0; i < size; i ++) {
				// Ht += W.slice(ims[i]).t() * Vt.col(i);
				Ht += W_user.slice(i).t() * Vt.col(i);
			}
			Ht = 1.0 / (1 + exp(-Ht));
			

			// negative phase: H -> V
			for (int i = 0; i < size; i++) {
				// Vt.col(i) = exp(BV.col(ims[i]) + W.slice(ims[i]) * Ht);
				Vt.col(i) = exp(BV.col(ims[i]) + W_user.slice(i) * Ht);
			}

			// Normalize Vt -> sum_k (Vt(k, i)) = 1
			Vt = normalise(Vt, 1);

		}


		// update BH
		BH += lrate * (H0 - Ht);



		// update B
		// update BV
		// update A
		mat B_old = B;
		for (int i = 0; i < size; i++) {
			mat HV_diff = (V0.col(i) * H0.t() - Vt.col(i) * Ht.t());
			BV.col(ims[i]) += lrate * (V0.col(i) - Vt.col(i));
			B += lrate * A.slice(ims[i]).t() * HV_diff;
			A.slice(ims[i]) += lrate * HV_diff * B_old.t();
		}


	}




	vector<float> predict_array(const record_array &rcd_array, const record_array &helper_array, unordered_map<unsigned int, int*> &predict_map, unordered_map<unsigned int, int*> &helper_map) {
		vector<float>results(rcd_array.size);
		int users[BATCH_SIZE];

		int thread_id = 0;
		int batch_id = 0;
		int* test_ids;
		for (int user_id = 1; user_id < N; user_id++) {
			unordered_map<unsigned int, int*>::const_iterator test_ids_iter = predict_map.find(user_id);
			if (test_ids_iter != predict_map.end()) {

				users[thread_id] = user_id;
				thread_id++;

				if (thread_id == BATCH_SIZE) {
#pragma omp parallel for num_threads(NUM_THREADS)
					for (int t = 0; t < thread_id; t++) {
						predict_user(users[t], rcd_array, helper_array, predict_map, helper_map, results);
					}

					thread_id = 0;
					batch_id++;
				}

			}
		}

		// user == N
		if (thread_id != 0) {
#pragma omp parallel for num_threads(NUM_THREADS)
			for (int t = 0; t < thread_id; t++) {
				predict_user(users[t], rcd_array, helper_array, predict_map, helper_map, results);
			}		
		}

		return results;
	}


	void predict_user(int user, const record_array &rcd_array, const record_array &helper_data, unordered_map<unsigned int, int*> &predict_map, unordered_map<unsigned int, int*> &helper_map, vector<float> &results) {

		vec Vum(K);
		ivec scores = linspace<ivec>(1, 5, 5);
		int* test_ids = predict_map[user];
		int* train_ids = train_map[user];

		int test_start = test_ids[0];
		int test_end = test_ids[1];
		int train_start = train_ids[0];
		int train_end = train_ids[1];


		// positive phase to compute Hu
		vec Hu = BH;
		for (int f = 0; f < F; f++) {
			for (int u = train_start; u < train_end; u++) {
				
				record r_train = ptr_train_data->data[u];
				unsigned int k = int(r_train.score) - 1;
				
				// double w = W(k, f, r_train.movie);
				double w  = 0;
				for (int c = 0; c < C; c++) {
					w += A(k, c, r_train.movie) * B(c, f);	
				}
				Hu(f) += w;
			}
		}
		Hu = 1.0 / (1 + exp(-Hu));

		// negative phase to predict score
		for (int u = test_start; u < test_end; u++) {
			record r_test = rcd_array.data[u];
			Vum = normalise( exp(BV.col(r_test.movie) + A.slice(r_test.movie) * B * Hu), 1);

			results[u] = dot(Vum, scores);
		}
	}


	virtual float predict(const record & rcd) {

		return 0.0;
	}



	void predict_qual_results_to_file(const float prob_rmse, unsigned int iter_num) {
		cout << "predicting qual data ..." << endl;
		vector<float>results = predict_array(*ptr_qual_data, *ptr_test_data, qual_map, test_map);

		// store results
		string out_dir = "frbm_results/";
		string rbm_out_name_pre;
		ostringstream convert;
		convert << prob_rmse << "_lrate" << this->lrate << "_F" << this->F << "_C" << this->C << "_iter" << iter_num;
		rbm_out_name_pre = out_dir + convert.str();
		string rbm_out_name = rbm_out_name_pre;

		for (int file_idx = 1; file_exists(rbm_out_name.c_str()); file_idx++) {
			rbm_out_name = rbm_out_name_pre + "_idx" + to_string(file_idx);
		}

		cout << "write to file: " << rbm_out_name << endl;

		ofstream rbm_out_file;
		rbm_out_file.open(rbm_out_name);
		for (int i = 0; i < ptr_qual_data->size; i++) {
			rbm_out_file << results[i] << endl;
		}
		rbm_out_file.close();
	}

	void write_prob_results_to_file(vector<float> results, const float prob_rmse, unsigned int iter_num) {

		// store results
		string out_dir = "frbm_results/";
		string rbm_out_name_pre;
		ostringstream convert;
		convert << "prob_" << prob_rmse << "_lrate" << this->lrate << "_F" << this->F << "_C" << this->C << "_iter" << iter_num;
		rbm_out_name_pre = out_dir + convert.str();
		string rbm_out_name = rbm_out_name_pre;

		for (int file_idx = 1; file_exists(rbm_out_name.c_str()); file_idx++) {
			rbm_out_name = rbm_out_name_pre + "_idx" + to_string(file_idx);
		}

		cout << "write to file: " << rbm_out_name << endl;

		ofstream rbm_out_file;
		rbm_out_file.open(rbm_out_name);
		for (int i = 0; i < ptr_qual_data->size; i++) {
			rbm_out_file << results[i] << endl;
		}
		rbm_out_file.close();
	}
};



unordered_map<unsigned int, int*> make_pre_map(const record_array &record_data) {
	unordered_map<unsigned int, int*> record_map;

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
			
			cur_user = this_data.user;
			cur_start = i;
		}
	}
	user_ids = new int[2];
	user_ids[0] = cur_start;
	user_ids[1] = record_data.size;
	record_map[cur_user] = user_ids;

	cout << "number of users = " << record_map.size() << endl;

	return record_map;
}



int main () {


	unsigned int ITER_NUM = 30;
	

	// string train_file_name = "../../../data/mini_main.data";
	// string test_file_name = "../../../data/mini_prob.data";
	// string qual_file_name = "../../../data/mini_prob.data"; // TODO: Change this name!!!
	string train_file_name = "../../../data/main_data.data";
	string test_file_name = "../../../data/prob_data.data";
	string qual_file_name = "../../../data/qual_data.data";
	
	record_array train_data;
	record_array test_data;
	record_array qual_data;
	train_data.load(train_file_name.c_str());
	cout << "finish loading " << train_file_name << endl;
	test_data.load(test_file_name.c_str());
	cout << "finish loading " << test_file_name << endl;
	qual_data.load(qual_file_name.c_str());
	cout << "finish loading " << qual_file_name << endl;


	basic_rbm rbm;
	rbm.ptr_train_data = &train_data;
	rbm.ptr_test_data = &test_data;
	rbm.ptr_qual_data = &qual_data;

	rbm.train_map = make_pre_map(train_data);
	rbm.test_map = make_pre_map(test_data);
	rbm.qual_map = make_pre_map(qual_data);



	rbm.fit(train_data, ITER_NUM);

	vector<float>results = rbm.predict_list(test_data);
	float prob_rmse = RMSE(test_data, results);
	cout << "RMSE: " << prob_rmse << endl;

	if (prob_rmse < 0.93) {
		rbm.predict_qual_results_to_file(prob_rmse, ITER_NUM);
	}


}





#endif