# RBM Report

### Sumamary

The project / repository is an implementation of restricted Boltzmann model (RBM) on GPU. Because of the low dependence among users, this model is, although not very easy to implement, highly parallelizable. I applied this algorithm on a part of the Netflix Challenge dataset. It helps cut the running time from the unit of minutes on CPU to the unit of seconds on GPU. 
This is a solo project in 3 weeks long.

### Background

The RBM is essentially a one layer neural network with an extra biased unit. It uses a latent factor approach to represent items with a few hidden factors. Generally speaking, a RBM model will take a set of binary vectors as inputs, and then use a gradient descent based approach , and iteratively reveal the relationship between the factors and movies. Then based on the previous set of inputs, the model could predict the new inputs based on activated factors.Specifically, in the case of user movie-ratings, the movies will be the visible layer of the model, and their hidden factors will be things such as fantasy, Hollywood movie, horror, etc. Then the biases connecting to movies and factors could be interpreted as the inclinations that a movie will get a specific score and that a factor will be activated. The RBM is recently growing popular and has been proved to have great performance on predictions of user specific applications. However, this model is extremely computationally expensive. In the example of Netflix dataset, which contains around 10 million data points, we need a single RBM model for every user, and the training of each model requires lots of matrix multiplications with dimensions of a few thousand times a few hundred. It makes the implementation on large dataset prohibitively slow in the case of CPU. Fortunately, the training of each user is almost independent of each other. So by taking advantage of this characteristic, we could implement the model on GPU and get a performance boost.


### Approach

The project could be baiscally divided into three blocks: `contrastive divergence on training`, `weights updates` and `predict results `.

#### Contrastive Divergence

As stated in the background section, I basically treat every user as a unit, and train them independently in different threads. Because of the memory limitation, I preprocessed the original data into batches and applied the kernel to batches sequentially.

Here is the type signature of the kernel:

~~~c++
void trainKernel(int* train_vec_in_batch, int* movies_in_batch, int* ratings_in_batch,
		float* Vzeros, float* Vts, float* Hzeros, float* Hts,
		float* W, float* BV, float* BH, float* W_inc, float* BV_inc, float* BH_inc,
		int batch_size, int num_movies_in_this_batch, const int i_batch_start,
		const bool update_weights);
~~~

The first line of arguments are the data related to the training set.  The train\_vec specifies the starting and ending index of records for every user. The movies\_in\_batch and ratings\_in\_batch store the all the records in the batch to lookup. 

The second line of arguments are the visible and hidden units for training. 

The third line of arguments are the weight maticies to help compute the values of the units.

The fourth line of arguments are the batch-specific information.

The final argument is a boolean to choose whether update weights or not. It's useful because I used the partial of this training kernel in the prediction stage and in that case this flag will set to be false.

#### Weights Update

After the contrastive diverngence of a batch, we need to update the weights. There are three majory components needs to update: **W** (weight matrix), **BV** (bias on visible units) and **BH** (bias on hidden units). I wrote seperate kernels to update them. 

The update of first two components is fairly trivial. During training, I set up matrices called **W_inc** and **BV_inc** with same dimension of W and BV to store the weight increments of these matrices. The udpates are purely parallel, and the code is as below:

~~~c++
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
~~~


The update of **BH** is less trivial. In contrast to the previous two matrices, where different users almost access all different rows/columns of the matrix, all users will try to modify and update this short-length vector. So we have to pay special attention to the writing conflicts. Also we have to update the weights after the execution of a batch. So I create a matrix **BH_inc** of dimension *F x BATCH_SIZE* to store the weight increments of **BH** where F is the length of the vector **BH**. Then I used the `reduction` method to update BH. The optimization follows the instruction from a presentation by Mark Harris. Details could be found in the References section.

~~~c++
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

...
// calling the kernel everytime to udpate one element of BH
for (int j = 0; j < F; j++) {
	updateBH_kernel<<<BHblocks, threadsPerBlock, threadsPerBlock * sizeof(float)>>>
		(BH + j, BH_inc + j, lrate_BH, batch_size);
~~~


#### Predict Results

The last part of the code is to predict results. Unlike a lot of other algorithms, the prediction stage of RBM is by no means trivial. 
It requires almost the positive phase of contrastive divergence applied to the training data as stated above, and then the negative phase of contrastive divergence applied to the testing data.

By passing a flag of `update_weights` to the training kernel, I saved half of the code, and avoided some possible mistakes. Then I wrote the predicting kernel with type signature as below:

~~~c++
void predictKernel(int* test_vec_in_batch, int* test_movies_in_batch, int* test_ratings_in_batch,
			float* Hzeros, float* Vts,
			float* W, float* BV,
			int batch_size, const int num_test_movies_in_this_batch, const int i_test_batch_start,
			float* results_in_batch);
~~~

Most arguments here have the similar meaning as the train kernel arguments. The last argument here is pre-allocated space to store the predicted results.

### Results

#### Speed

The data that I used for benchmark is a subset of Netflix dataset, which contains around 92,000 user, 3,553 movies and a total of 4,150,000 data points. For GPU, I set the batch size to be 131,072 and hence 131 thousand models will be trained simultaneously.

Machine: CMS MX

* Device: GeForce GTX 780
* Compute Capacity: 3.5
* Memory Clock Rate (KHz): 3004000
* Memory Bus Width (bits): 384
* Peak Memory Bandwidth (GB/s): 288.384000
* Total Memory (bytes): 3,220,897,792

Averaged time spent per iteration in **training**:

Process        | Elapsed Time (ms) | Features
-------------  | ----------------- | -----
GPU            | 676.627           | 10
GPU            | 8958.46           | 100
CPU            | 2025.65           | 10
CPU            | 11020.4           | 100

Averaged time spent per iteration in **predicting**:
	
Process        | Elapsed Time (ms) | Features
-------------  | ----------------- | -----
GPU            | 141.693           | 10
GPU            | 1549.11           | 100
CPU            | 293.042           | 10
CPU            | 1671.64           | 100

It improves the speed from CPU, but the speed-up is not that impressive as expected. I think the major reason is that even though the users can be trained independently, the training process for each one is computationally heavy. The GPU has characteristics of high throughput, but also high latency. So while GPU takes advantage of parallelization, its speed is caught up by CPU because of the slowness in doing massive serialized computation within a thread.

#### Prediction Accuracy

The best model trained on GPU has 120 features, and after trained with 40 iterations, it achieved a RMSE of 0.907, which is 4.6% above water. This result matches decently with the result of the paper by Hinton. 

### Usage

First, you need to have the Netflix dataset, and then organize the recrods into sturcture as below:

~~~c++
class record {

public:
	unsigned int user;
	unsigned int movie;
	unsigned int date;
	float score;
};
~~~

Then store the file into binary format, and you are good to go. More details of data storage are in the class `record_array` in the file `rbm.cc`. 

The Makefile is written. So it's easy to compile and run the program.

### Other Thoughts

**A good structure is half success**

The RBM is a reasonably complex model to implement, and there are lots of parameters needed to be configured. When I wrote the kernel declaration, I broke down the statement into multiple lines:

~~~c++
void train(int* train_vec_in_batch, int* movies_in_batch, int* ratings_in_batch,
	int* test_vec_in_batch, int* test_movies_in_batch, int* test_ratings_in_batch,
	float* Vzeros, float* Vts, float* Hzeros, float* Hts,
	float* W, float* BV, float* BH, float* W_inc, float* BV_inc, float* BH_inc,
	int batch_size, int num_movies_in_this_batch, const int i_batch_start,
	const int num_test_movies_in_this_batch, const int i_test_batch_start,
	const unsigned int M, const float lrate, const float lrate_BH,
	float* results_in_batch, const bool update_weights,
	int blocks, int threadsPerBlock);
~~~

The arguments on each line are correlated, and the name is very explicit. So I could easily match the argument with correct variable. Also, the variable type should be as close to its actual meaning as possible.

**CUDA is still immature**

When I initialized the weights as random numbers following normal distribution, I used the CuRand library, but there is a huge bug of the code that the function `curandGenerateNormal` takes only array with even length. Yes, there is 50% of chance of failure using the official CuRand library to generate normal distributed variabls.

Also, setting up `CuBlas` is extremely difficult. CuBlas is a library that helps on basic matrix operations. However, there are few documents existed and the process of setting up is painful. There is a file inside `helpers` folder called `Makefil-CuBlas` could be a reference to setup CuBlas in the future.

Because of the sorrow experience with CuRand, I discarded the CuBlas completely from the code implementation, and wrote nested for loops instead. There is definitely a tradeoff between these two choices. In the future, I hope to try CuBlas to see its performance.

### References

* [Restricted Boltzmann Machines for Collaborative Filtering, Hinton, et al]
(http://www.machinelearning.org/proceedings/icml2007/papers/407.pdf)
* [Netflix Challenge Official Website](http://www.netflixprize.com/)
* [Optimizing Parallel Reduction in CUDA, Harris.](https://docs.nvidia.com/cuda/samples/6_Advanced/reduction/doc/reduction.pdf)