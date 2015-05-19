#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <sstream>

#include <sys/stat.h>
#include <ctime>
#include <vector>
#include <omp.h>
#include <cuda_runtime.h>

// #include "cluster_cuda.cuh"

#define REVIEW_DIM 50

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
			system("pause");
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


void preprocess_data() {
	string train_file_name = "../../../data/mini_main.data";
	string test_file_name = "../../../data/mini_prob.data";
	string qual_file_name = "../../../data/mini_prob.data"; // TODO: Change this name!!!
	// string train_file_name = "../../../data/main_data.data";
	// string test_file_name = "../../../data/prob_data.data";
	// string qual_file_name = "../../../data/qual_data.data";

	record_array train_data;
	record_array test_data;
	record_array qual_data;


}


void rbm_core() {

}



int main(int argc, char** argv) {
  // int k = 5;
  // int batch_size = 32;
  int k = 50;
  int batch_size = 2048;



  if (argc == 1) {
  	;
    // cluster(cin, k, batch_size);
  } else if (argc == 2) {
    ifstream ifs(argv[1]);
    // stringstream buffer;
    // buffer << ifs.rdbuf();
    // cluster(buffer, k, batch_size);
  }
}




