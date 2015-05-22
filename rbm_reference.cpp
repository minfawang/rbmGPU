#include <armadillo>
#include <iostream>
#include "types.hpp"
// #include <shark/Unsupervised/RBM/BinaryRBM.h>


#ifndef __RBM_ESTIMATORS
#define __RBM_ESTIMATORS

using namespace arma;


bool isFuckedUp(double num) {
	return isnan(num) || isinf(num);
}


double sigma(double num) {
	return 1.0 / (1 + exp(-num));
}


class basic_rbm : public estimator_base {
public:

	cube W; // M * F * K
	mat BV; // K * M
	vec BH; // F
	// mat BH; // K * F

	unsigned int N;
	unsigned int M;
	unsigned int K;
	unsigned int F;
	unsigned int CD_K;
	double lrate; // learning rate


	record_array *ptr_test_data;
	record_array *ptr_train_data;




	basic_rbm() {
		K = 5;
		F = 60;
		M = 17770 / 10 + 1; // TODO: change M to be total number of movies
		N = 458293 / 10;

		W = randu<cube>(K, F, M) / 8.0;
		BV = randu<mat>(K, M) / 8.0;
		BH = randu<vec>(F) / 8.0;
		// BH = randu<mat>(K, F) / 8.0;


		CD_K = 5;
		lrate = 0.0003;


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
			// TEST CODE
			vector<float> results = predict_list(*ptr_test_data);
			cout << "RMSE: " << RMSE(*ptr_test_data, results) << endl;
			
			cout << "working on iteration " << iter_num << "..." << endl;

			unsigned int user_id = train_data.data[0].user;
			unsigned int start = 0;
			unsigned int end = 0;

			for (int i = 0; i < train_data.size; i++) {
				record r = train_data.data[i];
				if ((user_id != r.user) || i == train_data.size-1) {
					end = (i == train_data.size-1) ? (i + 1) : i;
					train((train_data.data+start), user_id, end - start);

					user_id = r.user;
					start = i;
				}
			}




			// store predicted data to file
			ofstream out_file;
		    out_file.open("test_coeff.txt");

		    // store W to file
		    for (int i = 0; i < M; i++) {
		    	for (int j = 0; j < F; j++) {
		    		out_file << W(0, j, i) << " ";
		    	}
		    	out_file << endl;
		    }

		    // store BV to file
		    for (int i = 0; i < M; i++) {
		    	for (int k = 0; k < K; k++) {
		    		out_file << BV(k, i) << " ";
		    	}
		    	out_file << endl;
		    }
		    
		    // store BH to file
		    for (int j = 0; j < F; j++) {
		    	out_file << BH(j) << endl;
		    }
		    
		    out_file.close();
		}


		cout << "finish training!" << endl;
		cout << "train data size: " << ptr_train_data->size << endl;
		cout << "test data size: " << ptr_test_data->size << endl;


	}





	vector<float> predict_list(const record_array & rcd_array) {
		// predicting stage
		unsigned int j = 0;
		unsigned int train_start = 0;
		unsigned int train_end = 0;
		unsigned int test_start = 0;
		unsigned int test_end = 0;
		unsigned int train_user = ptr_train_data->data[0].user;
		unsigned int test_user = ptr_test_data->data[0].user;

		vec Hu = zeros<vec>(F);
		vec Vum(K);
		ivec scores = linspace<ivec>(1, 5, 5);

		vector<float>results;
		results.resize(rcd_array.size);



		for (int i = 0; i < ptr_test_data->size; i++) {

			record r_test = ptr_test_data->data[i];

			if ((test_user != r_test.user) || i == ptr_test_data->size -1) {
				
				// make prediction of test_user for movies in the test set
				test_end = (i == ptr_test_data->size-1) ? (i + 1) : i;
				
				int u_size = test_end - test_start;

				// find train_start and train_end
				// record r_train = ptr_train_data->data[j];


				while (j < ptr_train_data->size) {
					record r_train = ptr_train_data->data[j];

					if (r_train.user < test_user) {
						train_start = j + 1;
					} else if (r_train.user > test_user) {
						break;
					}

					j++;
				}

				train_end = j;

				if (ptr_train_data->data[j-1].user == test_user) {

					// positive phase to compute Hu
					Hu = BH;
					for (int f = 0; f < F; f++) {
						// Hu(f) = BH(f);
						for (int u = train_start; u < train_end; u++) {
							
							record r_train = ptr_train_data->data[u];
							unsigned int k = int(r_train.score) - 1;
							
							double w = W(k, f, r_train.movie);
							Hu(f) += w;
						}

						// Hu(f) = sigma(Hu(f));
					}
					Hu = 1.0 / (1 + exp(-Hu));


					// negative phase to predict score
					for (int u = test_start; u < test_end; u++) {
						record r_test = ptr_test_data->data[u];
						Vum = normalise( exp(BV.col(r_test.movie) + W.slice(r_test.movie) * Hu), 1);
						results[u] = dot(Vum, scores);

					}


					/* way 2 */
					// for (int u = test_start; u < test_end; u++) {
					// 	record r_test = ptr_test_data->data[u];
					// 	vec rating_probs = zeros<vec>(K);
					// 	double predict_score = 0;


					// 	for (int k = 0; k < K; k++) {
					// 		rating_probs(k) = BV(k, r_test.movie);
					// 		for (int f = 0; f < F; f++) {
					// 			double w = W(k, f, r_test.movie); 
					// 			double h = Hu(f);
					// 			rating_probs(k) += w * h;
					// 		}

					// 		// TEST CODE
					// 		rating_probs(k) = exp(rating_probs(k));
					// 	}

					// 	// normalize rating_probs
					// 	// QUESTION: Is it possible for prob to be less than 0?
					// 	double sum_k = 0;
					// 	for (int k = 0; k < K; k++) {
					// 		sum_k += rating_probs(k);
					// 	}
					// 	for (int k = 0; k < K; k++) {
					// 		rating_probs(k) /= sum_k;
					// 	}

					// 	// update predict score by taking average
					// 	for (int k = 0; k < K; k++) {
					// 		predict_score += (k+1) * rating_probs(k);
					// 	}

					// 	// cout << predict_score << "  ";
					// 	results[u] = predict_score;

					// }

				} else {
					// TODO: predict all movies to be 3.5
					double predict_score;
					for (int u = test_start; u < test_end; u++) {
						predict_score = 3.6;
						results[u] = predict_score;
					}
				}

				train_start = j;


				test_start = i;
				test_user = r_test.user;
			}
		}

		// cout << "finish predicting!" << endl;


		// ceil and floor result
		for (int i = 0; i < ptr_test_data->size; i++) {
			if (results[i] > 5)
				results[i] = 5;
			else if (results[i] < 1) 
				results[i] = 1;
		}

	    return results;

	}



	virtual float predict(const record & rcd) {

		return 0.0;
	}




	void train(const record *data, unsigned int user_id, unsigned int size) {
		// initialization
		mat V0 = zeros<mat>(K, size);
		mat Vt = zeros<mat>(K, size);
		vec H0 = zeros<vec>(F);
		vec Ht = zeros<vec>(F);


		// set up V0 and Vt based on the input data.
		for (int i = 0; i < size; i++) {
			record r = data[i];
			V0(int(r.score)-1, i) = 1; // score - 1 is the index
			Vt(int(r.score)-1, i) = 1;

		}



		/*
		/////////////////// set up H0 by V -> H //////////////////
		H0(j) = sigma( BH(j) + sum_ik ( W(k, j, r.movie) * V0(k, i) ))
		*/

		/* way 1 */
		H0 = BH;
		for (int i = 0; i < size; i++) {
			H0 += W.slice(data[i].movie).t() * V0.col(i);
		}
		H0 = 1.0 / (1 + exp(-H0));

		/*  //////// way 2
		for (int j = 0; j < F; j++) {
			H0(j) = BH(j);
			for (int i = 0; i < size; i++) {
				record r = data[i];
				for (int k = 0; k < K; k++) {
					double w = W(k, j, r.movie);
					double v = V0(k, i);
					H0(j) += w * v;
				}
			}
			H0(j) = sigma(H0(j));
		}
		*/
		


		/////////////////// Do the contrastive divergence ///////////////////
		for (int n = 0; n < CD_K; n++) {

			////////////// positive phase: V -> H /////////
			Ht = BH;
			for (int i = 0; i < size; i ++) {
				Ht += W.slice(data[i].movie).t() * Vt.col(i);
			}
			Ht = 1.0 / (1 + exp(-Ht));


			 ///// way 2 //////

			// for (int j = 0; j < F; j++) {
			// 	Ht(j) = BH(j);

			// 	for (int i = 0; i < size; i++) {
			// 		record r = data[i];

			// 		for (int k = 0; k < K; k++) {
			// 			double w = W(k, j, r.movie);
			// 			double v = Vt(k, i);

			// 			Ht(j) += w * v;

			// 		}
			// 	}

			// 	Ht(j) = sigma(Ht(j));
			// }
			





			// negative phase: H -> V
			//TEST CODE
			// Vt = exp(BV.cols(mIndices) + W.slices(mIndices) * Ht);
			for (int i = 0; i < size; i++) {
				record r = data[i];

				Vt.col(i) = exp(BV.col(r.movie) + W.slice(r.movie) * Ht);

				// for (int k = 0; k < K; k++) {

				// 	double bv = BV(k, r.movie);
				// 	Vt(k, i) = bv;

				// 	for (int j = 0; j < F; j++) {
				// 		double h = Ht(j);
				// 		double w = W(k, j, r.movie);

				// 		Vt(k, i) += h * w;
				// 	}

				// 	Vt(k, i) = exp(Vt(k, i));
				// }
			}

			// TEST CODE
			// Normalize Vt -> sum_k (Vt(k, i)) = 1
			Vt = normalise(Vt, 1);
			/* way 2 */
			// for (int i = 0; i < size; i++) {
			// 	double sum_k = 0.0;

			// 	for (int k = 0; k < K; k++) 
			// 		sum_k += Vt(k, i);

			// 	for (int k = 0; k < K; k++) 
			// 		Vt(k, i) /= sum_k;
			// }
			

		}

		// update W
		for (int i = 0; i < size; i++) {
			record r = data[i];

			W.slice(r.movie) += lrate * (V0.col(i) * H0.t() - Vt.col(i) * Ht.t());
			/* way 2 */
			// for (int j = 0; j < F; j++) {
			// 	for (int k = 0; k < K; k++) {

			// 		W(k, j, r.movie) += lrate * (H0(j) * V0(k, i) - Ht(j) * Vt(k, i));

			// 	}
			// }
		}

		// update BH

		BH += lrate * (H0 - Ht);
		/* way 2 */
		// for (int j = 0; j < F; j++) {
		// 	BH(j) += lrate * (H0(j) - Ht(j));
		// }

		// update BV
		for (int i = 0; i < size; i++) {
			record r = data[i];

			BV.col(r.movie) += lrate * (V0.col(i) - Vt.col(i));
			/* way 2 */
			// for (int k = 0; k < K; k++) {
			// 	BV(k, r.movie) += lrate * (V0(k, i) - Vt(k, i));
			// }
		}

	}


};


int main () {
	string train_file_name = "../../data/mini_main.data";
	string test_file_name = "../../data/mini_prob.data";
	// string train_file_name = "../../data/main_data.data";
	// string test_file_name = "../../data/prob_data.data";
	
	record_array train_data;
	train_data.load(train_file_name.c_str());
	cout << "finish loading " << train_file_name << endl;


	basic_rbm rbm;

	rbm.ptr_train_data = &train_data;


	record_array test_data;
	test_data.load(test_file_name.c_str());
	cout << "finish loading " << test_file_name << endl;
	rbm.ptr_test_data = &test_data;


	unsigned int iter_num = 40;
	rbm.fit(train_data, iter_num);

	vector<float>results = rbm.predict_list(test_data);
	cout << "RMSE: " << RMSE(test_data, results) << endl;


	// store results
	ofstream rbm_out_file;
	rbm_out_file.open("test_rbm_out.txt");
	for (int i = 0; i < test_data.size; i++) {
		rbm_out_file << results[i] << endl;
	}
	rbm_out_file.close();

}





#endif