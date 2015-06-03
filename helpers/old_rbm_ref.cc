		/* Old version implementation */


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
